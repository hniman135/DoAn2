import shutil
import os
def changeMess (mess):
    # Copy the file
    source_file = 'C:/Users/vomin/Source/PycharmProjects/esp32/sms_esp/main/main1.c'
    destination_file = 'C:/Users/vomin/Source/PycharmProjects/esp32/sms_esp/main/main.c'

    # Copy the file
    shutil.copyfile(source_file, destination_file)

    print("File copied successfully.")
    # Open the file in read mode
    with open('C:/Users/vomin/Source/PycharmProjects/esp32/sms_esp/main/main.c', 'r') as file:
        # Read the file contents
        content = file.read()

    # Check if the word "hello" exists in the file
    if 'Hello' in content:
        # Replace "hello" with "HI"
        updated_content = content.replace('Hello', mess)

        # Open the file in write mode
        with open('C:/Users/vomin/Source/PycharmProjects/esp32/sms_esp/main/main.c', 'w') as file:
            # Write the updated content to the file
            file.write(updated_content)

        print("Replacement complete.")
    else:
        print("The word 'hello' was not found in the file.")

def sendMess():
    # Specify the directory path you want to change to
    new_directory = 'C:/Users/vomin/Source/PycharmProjects/esp32/sms_esp'

    # Change the current working directory
    os.chdir(new_directory)
    os.system('idf.py -p COM5 flash')