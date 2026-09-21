#!/usr/bin/env python3
"""
API 可用性测试脚本
运行: python test_apis.py
"""

import requests
import json
from config import Config  # 导入你的配置


def test_hkgai_llm():
    """测试 HKGAI LLM API"""
    try:
        url = f"{Config.HKGAI_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {Config.HKGAI_API_KEY}"}
        payload = {
            "model": Config.HKGAI_MODEL_ID,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, "成功: 收到响应", data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
    except requests.exceptions.RequestException as e:
        return False, f"失败: {str(e)}", None


def test_serper_search():
    """测试 Serper.dev 搜索引擎 API"""
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": Config.SERPAPI_KEY, "Content-Type": "application/json"}
        payload = {"q": "test query"}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, "成功: 收到搜索结果", len(data.get('organic', []))
    except requests.exceptions.RequestException as e:
        return False, f"失败: {str(e)}", None


def test_openweather():
    """测试 OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Beijing&appid={Config.OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, "成功: 收到天气数据", data.get('main', {}).get('temp', 'N/A')
    except requests.exceptions.RequestException as e:
        return False, f"失败: {str(e)}", None


def test_alpha_vantage():
    """测试 Alpha Vantage 金融 API"""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={Config.ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, "成功: 收到股票数据", list(data.get('Time Series (Daily)', {}).keys())[
                                           :1] if 'Time Series (Daily)' in data else 'N/A'
    except requests.exceptions.RequestException as e:
        return False, f"失败: {str(e)}", None


def test_baidu_map():
    """测试 Baidu Map API (距离矩阵示例)"""
    if Config.BAIDU_MAP_API_KEY == "demo_mode":
        return False, "跳过: 未配置密钥 (demo_mode)", None
    try:
        url = f"https://api.map.baidu.com/directionlite/v1/driving?origin=40.56,116.31&destination=40.55,116.30&ak={Config.BAIDU_MAP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return True, "成功: 收到地图数据", data.get('result', {}).get('routes', [{}])[0].get('distance', 'N/A')
    except requests.exceptions.RequestException as e:
        return False, f"失败: {str(e)}", None


def main():
    tests = [
        ("HKGAI LLM", test_hkgai_llm),
        ("Serper Search", test_serper_search),
        ("OpenWeather", test_openweather),
        ("Alpha Vantage", test_alpha_vantage),
        ("Baidu Map", test_baidu_map),
    ]

    print("=== API 可用性测试报告 ===")
    all_passed = True
    for name, test_func in tests:
        success, message, sample = test_func()
        status = "✅ 通过" if success else "❌ 失败"
        print(f"\n{name}: {status}")
        print(f"  详情: {message}")
        if sample:
            print(f"  示例数据: {sample}")
        all_passed = all_passed and success

    print(f"\n=== 总体结果: {'全部通过' if all_passed else '部分失败，请检查密钥/网络'} ===")


if __name__ == "__main__":
    main()