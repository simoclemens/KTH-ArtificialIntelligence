import sys


def matrix_prod(mat1,mat2):

    res = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                res[i][j] += mat1[i][k] * mat2[k][j]

    return res

#lines = open(sys.argv[1]).readlines()
lines = sys.stdin.readlines()

# transition matrix
A = []
elems1 = lines[0].strip().split(" ")
n_row1 = int(elems1[0])
n_col1 = int(elems1[1])
numbers1 = elems1[2:]

for i in range(n_row1):
    tmp = []
    for j in range(n_col1):
        tmp.append(float(numbers1[i*n_col1+j]))
    A.append(tmp)

# observations matrix
B = []
elems2 = lines[1].strip().split(" ")
n_row2 = int(elems2[0])
n_col2 = int(elems2[1])
numbers2 = elems2[2:]

for i in range(n_row2):
    tmp = []
    for j in range(n_col2):
        tmp.append(float(numbers2[i*n_col2+j]))
    B.append(tmp)

# initial probabilities
elems3 = lines[2].strip().split(" ")
p0 = [[float(i) for i in elems3[2:]]]

x = matrix_prod(p0, A)

res = matrix_prod(x,B)
out = ""

out += " " + str(len(res))
out += " " + str(len(res[0]))

for p in res[0]:
    out += " " + str(p)

print(out)