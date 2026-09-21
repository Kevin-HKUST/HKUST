# server.py
import sys
import os
import tempfile
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.workflow_engine import WorkflowEngine
from multimodal.multimodal_router import MultimodalRouter
from multimodal.docx_processor import DocxProcessor
from core.llm_client import llm_client
import uvicorn
import time
import re

# Import batch processing functions from main.py（确保使用修改后的load_test_questions）
from main import load_test_questions, calculate_quality

app = FastAPI()

# Initialize multimodal router
multimodal_router = MultimodalRouter()
docx_processor = DocxProcessor()

# Ultimate CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Custom JSON serializer
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        else:
            return str(obj)


# Initialize workflow engine
try:
    engine = WorkflowEngine()
    print("✅ WorkflowEngine initialized successfully (empty vector store does not affect basic functions)")
except Exception as e:
    print(f"❌ WorkflowEngine initialization warning: {str(e)}")
    engine = None


# Global response header middleware
@app.middleware("http")
async def add_universal_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


# Handle OPTIONS preflight requests
@app.options("/query")
@app.options("/batch_query")
async def handle_options():
    return JSONResponse(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        }
    )


# Normal query endpoint (single question/file)
@app.post("/query")
async def query(question: str = Form(...), file: UploadFile = File(None)):
    response_data = {
        "success": False,
        "answer": "",
        "confidence": 0.0,
        "error": "",
        "debug": {}
    }

    if not engine:
        response_data["error"] = "Backend engine initialization failed, please restart server.py"
        return JSONResponse(response_data)

    temp_file_path = None
    extracted_text = ""
    try:
        if file is not None:
            file_ext = os.path.splitext(file.filename)[1].lower() or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(await file.read())
                temp_file_path = tmp.name
            response_data["debug"]["temp_file_path"] = temp_file_path

            router_result = multimodal_router.load(temp_file_path)
            if "error" in router_result:
                response_data["error"] = f"File processing failed: {router_result['error']}"
                return JSONResponse(response_data)

            extracted_text = router_result.get("text", "").strip()
            if extracted_text:
                response_data["debug"]["extracted_text_length"] = len(extracted_text)
            else:
                response_data["debug"]["warning"] = "No text extracted from file"

        if extracted_text:
            formatted_query = (
                f"Question: {question}\n\n"
                f"[Extracted from file ({router_result.get('type', 'unknown')}):]\n"
                f"{extracted_text}"
            )
        else:
            formatted_query = question

        response_data["debug"]["formatted_query"] = formatted_query[:150]

        workflow_result = engine.run(formatted_query)

        final_answer = workflow_result.get("final_answer", {})
        response_data["success"] = True
        response_data["answer"] = final_answer.get("answer", "No valid answer retrieved")
        response_data["confidence"] = round(final_answer.get("confidence", 0.0), 2)
        response_data["debug"]["processing_time"] = round(workflow_result.get("processing_time", 0.0), 3)

    except Exception as e:
        error_msg = str(e)
        response_data["error"] = f"Backend processing failed: {error_msg[:100]}"
        response_data["debug"]["exception"] = error_msg
        print(f"❌ Request processing failed：{error_msg}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                response_data["debug"]["temp_file_cleaned"] = True
            except Exception as e:
                response_data["debug"]["temp_file_cleaned"] = f"Failed: {str(e)}"

    try:
        json.dumps(response_data, cls=JSONEncoder)
    except Exception as e:
        response_data["debug"] = "Partial debug info removed due to serialization failure"
        response_data["error"] = f"{response_data['error']} | JSON serialization warning: {str(e)}"

    return JSONResponse(response_data)


# Batch query endpoint (process all questions in document)
@app.post("/batch_query")
async def batch_query(file: UploadFile = File(...), question: str = Form("")):
    response_data = {
        "metadata": {
            "input_file": "",
            "processing_start_time": "",
            "processing_end_time": ""
        },
        "stats": {
            "total_questions": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_time_seconds": 0.0,
            "avg_time_per_question_seconds": 0.0,
            "success_rate_percent": 0.0
        },
        "questions": [],
        "error": ""
    }

    if not engine:
        response_data["error"] = "Backend engine initialization failed, please restart server.py"
        return JSONResponse(response_data)

    temp_file_path = None
    try:
        # Save uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1].lower() or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            temp_file_path = tmp.name

        response_data["metadata"]["input_file"] = file.filename
        response_data["metadata"]["processing_start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()

        # Extract questions from document（使用修改后的load_test_questions）
        questions = load_test_questions(temp_file_path)
        if not questions:
            response_data[
                "error"] = "No valid questions extracted from the document (supports Chinese/English/duplicate questions)"
            return JSONResponse(response_data)

        # Process each question（逐个处理，保留所有问题）
        answers = []
        total_processing_time = 0.0
        success_count = 0

        for idx, q in enumerate(questions, 1):
            print(f"Batch processing question {idx}/{len(questions)}: {q[:50]}...")
            q_start_time = time.time()

            # Execute workflow
            result = engine.execute_workflow(q)
            final_answer = result.get("final_answer", {})
            answer_content = final_answer.get("answer", "").strip()
            confidence = final_answer.get("confidence", 0.0)

            # Calculate quality score and success status
            processing_time = time.time() - q_start_time
            quality_score = calculate_quality(answer_content)
            is_success = quality_score >= 0.5

            # Accumulate statistics
            total_processing_time += processing_time
            if is_success:
                success_count += 1

            # Save individual result（保留原始问题文本）
            answers.append({
                "question_id": idx,
                "question": q,
                "answer": answer_content,
                "confidence": round(confidence, 2),
                "quality_score": round(quality_score, 2),
                "processing_time_seconds": round(processing_time, 3),
                "is_success": is_success
            })

        # Calculate final statistics
        total_questions = len(questions)
        response_data["stats"] = {
            "total_questions": total_questions,
            "success_count": success_count,
            "failure_count": total_questions - success_count,
            "total_time_seconds": round(time.time() - start_time, 3),
            "avg_time_per_question_seconds": round(total_processing_time / total_questions,
                                                   3) if total_questions > 0 else 0.0,
            "success_rate_percent": round((success_count / total_questions) * 100, 2) if total_questions > 0 else 0.0
        }
        response_data["questions"] = answers
        response_data["metadata"]["processing_end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        error_msg = str(e)
        response_data["error"] = f"Batch processing failed: {error_msg}"
        print(f"❌ Batch processing failed：{error_msg}")

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"❌ Failed to clean up temp file: {str(e)}")

    return JSONResponse(response_data)


if __name__ == "__main__":
    print("🚀 Backend service starting...")
    print("Access URL: http://localhost:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        use_colors=True
    )