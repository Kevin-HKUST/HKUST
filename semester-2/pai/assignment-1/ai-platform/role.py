class Role:
    """
    Class representing a role in the AI model service system.

    Role is a base class which should be inherited from to provide additional functionality.
    A role consists of the name of the role and the set of permissions for this role 
    (represented as str objects).
    By default, a role has no permissions.
    """
    def __init__(self, name: str, permissions: list[str] | None = None):
        self.__name = name
        self.__permissions = set(permissions) if permissions else set()

    def get_name(self) -> str:
        """Gets the name of the role."""
        return self.__name

    def has_permission(self, perm: str) -> bool:
        """Checks whether the role has a given permission."""
        return perm in self.__permissions

    def update_permissions(self, new_perms: list[str]) -> None:
        """Adds the given permissions to the existing set of permissions."""
        self.__permissions.update(new_perms)


class StandardRole(Role):
    """Standard role with 'predict' and 'view_info' permissions."""
    def __init__(self, name: str = "Standard"):
        super().__init__(name, ["predict", "view_info"])


class AdminRole(StandardRole):
    """Admin role inheriting Standard permissions, plus 'train' and 'delete_model'."""
    def __init__(self):
        super().__init__("Admin")
        self.update_permissions(["train", "delete_model"])


class User:
    """
    Class representing a user in the AI model service system.

    A user has a username, a role, the remaining quota for predictions, 
    and the list of timestamps of the request history.

    :param username: The name of the user
    :param role: The role of the user
    :param quota: The remaining quota for predictions
    """
    def __init__(self, username: str, role: Role, quota: int = 100):
        self.__username = username
        self.__role = role
        self.__quota = quota
        self.__request_history = []

    def get_username(self) -> str:
        """Gets the name of the user."""
        return self.__username

    def get_role(self) -> Role:
        """Gets the role of the user."""
        return self.__role

    def get_quota(self) -> int:
        """Gets the remaining quota of the user."""
        return self.__quota

    def get_request_history(self) -> list[int]:
        """Gets the timestamps representing the request history of the user."""
        return self.__request_history

    def subtract_quota(self, val: int) -> None:
        """
        Subtracts the input value from the quota of the user. 
        Note that no validation is performed.
        
        :param val: The quota to subtract
        """
        self.__quota -= val

    def add_history(self, time: int) -> None:
        """Adds this timestamp to the history record."""
        self.__request_history.append(time)

    def prune_history(self, before: int) -> None:
        """
        Prunes all history records before the given time (exclusive).
        That is, if `before` exists as a timestamp in the history, do NOT remove it.

        :param before: The cutoff to remove history records before
        """
        self.__request_history = [t for t in self.__request_history if t >= before]

    def get_history_length(self) -> int:
        """Gets the length of the history."""
        return len(self.__request_history)

if __name__ == "__main__":
    print("This is a library. Please do not run this file directly.")