n=int(input("Enter number of elements:"))
d=input("Enter the element:")
data= list(int(num) for num in d.split())[0:n]
ln=len(data)
s=sum(data)
mean=s/ln
print("Mean is",mean)

data=sorted(data)
if ln%2==0:
    med1=data[ln//2]
    med2=data[ln//2-1]
    median= (med1+med2)/2
else:
    median=data[ln//2]
print("Median is:",median)

def find_mode(data):
    count_dict={}
    for num in data:
        if num in count_dict:
            count_dict[num] +=1
        else:
            count_dict[num]=1
    max_count=max(count_dict.values())
    mode=[key for key,value in count_dict.items() if value==max_count]
    return mode
print("Mode is /are:",find_mode(data))



varience=sum((x-mean)**2 for x in data)/ln
print("Varience is:",varience)


std=varience**0.5
print("Standered deviaton is:",std)


