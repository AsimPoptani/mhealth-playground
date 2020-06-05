import glob

# Warning takes 7 seconds to load
# Gives data in a dictionary on the mhealth dataset: [ {id:id,data: 2DArrayOfRawData ,file_location:fileLocation},{id:id,data: 2DArrayOfRawData ,file_location:fileLocation}]
def mhealth_get_dataset(dir_to_files = './mhealth-data/'):
    ''' Returns a dictionary of subjects.

    This DocString is a test to make sure git is working! 
    '''

    # get our file uris
    files_uri=glob.glob(dir_to_files+'*.log')

    # Keep track of data
    subjects=[]

    # Read data and put data into subjects and use enumerate to id
    for linear_identifier,file_uri in enumerate(files_uri):

        # Open the file in read only mode
        mhealth_file=open(file_uri,'r')
        
        # Keep track of file data
        mhealth_data=[]
        # Loop through the file 
        for line in mhealth_file.readlines():
            
            # Get the floating value of each column for a row and append to mhealth_data
            row=list(map(float,line.split()))
            mhealth_data.append(row)        
        mhealth_file.close()
        
        # Build subject
        subject = {
            'id':linear_identifier,
            'data': mhealth_data,
            'file_location':file_uri
        }
        # Append subject
        subjects.append(subject)
    return subjects



if __name__ == "__main__":
    # Dirty testing 
    import time
    print("--- start ---")
    start_time = time.time()

    lst = mhealth_get_dataset()
    print("length of lst: ", len(lst))

    print("--- end ---")
    print("--- %s seconds ---" % (time.time() - start_time))