# Give data in a dictionary : { subjectX : 2D array of data, subjectY : 2D array of data}
def mHealth_get_dataset():
    # Root dir
    dir_to_files = './mlhealth-data/'

    # get our files
    files=[]
    # Go through append files
    for index in range(1,11):
        files.append(f"mHealth_subject{index}.log")

    # Read data
    for uri_name in files:
        uri=dir_to_files+uri_name
        mlhealth_file=open(uri,'r')
        print(mlhealth_file.read())
        mlhealth_file.close()
