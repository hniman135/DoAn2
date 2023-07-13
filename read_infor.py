
def get_info(input):
    with open('C:/Users/vomin/Source/PycharmProjects/object_detection/member.txt', 'r') as file:
        for line in file:
            data = line.strip().split(': ')
            if data[3] == input:
                return data[3],int(data[2])
    return data[3],None

def pay(plate_num):
    lines = []
    with open('C:/Users/vomin/Source/PycharmProjects/object_detection/member.txt', 'r') as file:
        for line in file:
            data = line.strip().split(': ')
            if plate_num == data[3]:
                payment = int(data[2]) - 10
                line = line.replace(data[2], str(payment), 1)
            lines.append(line)
    with open('C:/Users/vomin/Source/PycharmProjects/object_detection/member.txt', 'w') as file:
        file.writelines(lines)
    if payment:
        return payment
    else:
        return None

def checker(plate):
    status_customer = 0
    new_pay = 0
    plate, balance = get_info(plate)
    if (balance != None):
        status_customer = 1
        new_pay = pay(plate)
    return status_customer, new_pay
