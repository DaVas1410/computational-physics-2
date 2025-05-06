class Student:
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError("Name must be a string")
        if not new_name:
            raise ValueError("Name cannot be empty")
        self._name = new_name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, new_age):
        if not isinstance(new_age, int):
            raise TypeError("Age must be an integer")
        if new_age <= 0 or new_age > 110:
            raise ValueError("Age must be a positive integer and within a reasonable range")
        self._age = new_age

    @classmethod
    def setup_class(cls):
        # Set up class-level resources (e.g., database connections)
        print("Setting up shared_data.")
        cls.shared_data = "Shared data in class"

    @classmethod
    def teardown_class(cls):
        # Tear down class-level resources
        print("Tearing down shared_data.")
        # Clean up anything created in setup_class (standard for writing tests).
        del cls.shared_data
