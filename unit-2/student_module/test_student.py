import pytest
from student import Student

class TestStudent:

    #call the class methods from the student class:
    
    @classmethod
    def setup_class(cls):
        Student.setup_class()

    @classmethod
    def teardown_class(cls):
        Student.teardown_class()

    # Validation tests: (8 routines)

    # Tests the initialization of a Student object with valid name and age.
    def test_student_initialization(self):
        student = Student("Alice", 20)
        assert student.name == "Alice"
        assert student.age == 20

    # Tests the ability to change a Student's name using the name setter.
    def test_student_name_setter(self):
        student = Student("Alice", 20)
        student.name = "Bob"
        assert student.name == "Bob"

    # Tests the ability to change a Student's age using the age setter.
    def test_student_age_setter(self):
        student = Student("Alice", 20)
        student.age = 21
        assert student.age == 21

    # Tests that a ValueError is raised when attempting to set an empty name.
    def test_student_name_validation_raises(self):
        student = Student("Alice", 20)
        with pytest.raises(ValueError):
            student.name = ""

    # Tests that a ValueError is raised when attempting to set an unreasonable (negative) age.
    def test_student_age_validation_raises(self):
        student = Student("Alice", 20)
        with pytest.raises(ValueError):
            student.age = -10 # Unreasonable age

    def test_student_age_type_validation_raises(self):
        student = Student("Alice", 20)
        # This ttempts to assign the string value "twenty" to the age property of the student object.
        with pytest.raises(TypeError):
            student.age = "twenty"

    # Verify that your Student class correctly raises a TypeError when you attempt to assign a non-string value to the name property.
    def test_student_name_type_validation_raises(self):
        student = Student("Alice",20)
        with pytest.raises(TypeError):
            student.name = 20

    # Checking for the Attribute. hasattr(Student, "shared_data") checks if the Student class has an attribute named "shared_data".
    def test_class_resource_exists(self):
        assert hasattr(Student, "shared_data")
