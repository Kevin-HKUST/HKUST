from role import *
import functools
import time

"""
Provided to store data relevant to a user session.
"""
__current_session: dict[str, User| None] = {"user": None}

def set_current_user(u: User) -> None:
    """Sets the current user."""
    __current_session["user"] = u

def require_permission(perm: str):
    """
    Checks whether the user has a given permission.

    If the current user does not exist, or if they do not have the permission requested,
    raise `PermissionError`.

    Otherwise, return the result value of the wrapped function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user = __current_session["user"]
            if user is None or not user.get_role().has_permission(perm):
                raise PermissionError(
                    f"User does not have the '{perm}' permission."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def charge_usage(base_cost: int):
    """
    Charges the user by computing the dynamic cost of inference based on the image size.

    First, check if the current user exists. If not, return.
    Then, get the image size (`image_size_kb`) from the `kwargs`, 
    and compute the total cost as the base cost plus the size multiplied by 0.1.

    If the user does not have sufficient quota, raise `ValueError`.
    Otherwise, run inference with the model. If the model successfully runs inference
    and does not crash with an error, subtract the quota, print a message, and return.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user = __current_session["user"]
            if user is None:
                return

            image_size_kb = kwargs.get("image_size_kb", 0)
            total_cost = base_cost + image_size_kb * 0.1

            if user.get_quota() < total_cost:
                raise ValueError(
                    f"Insufficient quota. Required: {total_cost}, "
                    f"Available: {user.get_quota()}"
                )

            result = func(*args, **kwargs)

            user.subtract_quota(total_cost)
            # Please print the message in the following format:
            print(f"[Billing] Deducted {total_cost} quota from the user. Remaining quota: {user.get_quota():.2f}")
            return result
        return wrapper
    return decorator

def rate_limit(max_calls: int, period_seconds: int):
    """
    Implements sliding-window based rate limiting.

    First, check if the current user is not `None`. If not, return.
    Then, get the current time as an integer by using `time.time()` and performing the appropriate conversion.
    Prune the user's request history by removing all function history timestamps `current_time - period_seconds`.
    Check the current number of requests remaining in the request history.
    If the number of requests exceed `max_calls`, raise a `PermissionError`.
    Otherwise, call the wrapped function, append the current timestamp to the request history, and return the result.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user = __current_session["user"]
            if user is None:
                return

            current_time = int(time.time())

            user.prune_history(current_time - period_seconds)

            if user.get_history_length() >= max_calls:
                raise PermissionError(
                    f"Rate limit exceeded: max {max_calls} calls "
                    f"per {period_seconds}s."
                )

            result = func(*args, **kwargs)
            user.add_history(current_time)
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    print("This is a library. Please do not execute it directly.")