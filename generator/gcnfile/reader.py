import numpy as np
size = 34
def preprocessinput(file_name,write_name):
    key2array = {}
    for i in range(1, size+1):
        key2array[i] = []
    with open(file_name) as file:
        for line in file:
            array = []
            line = line.split(",")
            # print(len(line[1]))
            line[1] = line[1][:-1]
            if len(line) == 2:
                if int(line[1]) not in key2array[int(line[0])]:
                    key2array[int(line[0])].append(int(line[1]) )
                if int(line[0]) not in key2array[int(line[1])]:
                    key2array[int(line[1])].append(int(line[0]) )
    adj = [[0 for i in range(size)] for i in range(size)]
    # print(adj[1][1])
    for i in range(1,size+1):
        for x in key2array[i]:
            adj[i-1][int(x)-1] += 1
    # print(adj)
    # exit(5)
    for i in range(1,size+1):
        print(" ".join([str(x) for x in adj[i-1]]))



def read(file_name):
    # inputMatrix = np.zeros((34,34))
    with open(file_name) as file:
        matrix = []
        for line in file:
            array = []
            line = line.split(" ")
            if line[len(line)-1].endswith("\n"):
                line[len(line)-1] = line[len(line)-1][:-1]
            for vertex in line:
                array.append(float(vertex))
            matrix.append(array)
    print("origin",matrix)
    return matrix

def readoutput():
    return
def read_data(input, label,current_dir):
    preprocessinput(current_dir + "/gcnfile/" + "plain.data",current_dir + "/gcnfile/" + input)
    input_dir = current_dir + "/gcnfile/" + input
    label_dir = current_dir + "/gcnfile/" + label
    return read(input_dir), read(label_dir)