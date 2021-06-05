class SchoolMember:
    '''代表了学校中的任何一个成员'''
    def __init__(self, name, age):
        self.name = name
        self.age = age
        #print('(Initialized SchoolMember: {})'.format(self.name))

    '''def tell(self):
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")'''


class Teacher(SchoolMember):
    '''表征一个老师'''
    def __init__(self, name, age, salary):
        super().__init__(name,age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        print('Name:"{}" Age:"{}" Salary:"{:d}"'.format(self.name,self.age,self.salary))


class Student(SchoolMember):
    '''表征一个学生'''
    def __init__(self, name, age, marks):
        super().__init__(name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        print('Name:"{}" Age:"{}" Marks:"{:d}"'.format(self.name,self.age,self.marks))

t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)

# 输出一个空行
print()

members = [t, s]
for member in members:
    # 所有的老师和学生都可用
    member.tell()