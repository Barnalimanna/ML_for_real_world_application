class cal():
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def add(self):
        return self.a+self.b
    def sub(self):
        return self.a-self.b
    def mul(self):
        return self.a*self.b
    def div(self):
        return self.a/self.b
    def exp(self):
        return self.a**self.b
    def intdiv(self):
        return (self.a//self.b)
a=int(input("Enter first number:"))
b=int(input("Enter second number:"))
obj=cal(a,b)
choice=1
while choice!=0:
        print("0: for exit")
        print("1: for addition")
        print("2: for subtraction")
        print("3: for multiplication")
        print("4: for division")
        print("5: for exponentiation")
        print("6: for integer division")
        choice=int(input("Enter choice: "))
        if choice==1:
            print("Result: ",obj.add())
        if choice==2:
            print("Result: ",obj.sub())
        if choice==3:
            print("Result: ",obj.mul())
        if choice==4:
            print("Result: ",round(obj.div(),3))      
        if choice==5:
            print("Result: ",obj.exp())
        if choice==6:
            print("Result: ",obj.intdiv())
        if choice==0:
            print("Exiting")
        else:
            print("Invalid choice!")

