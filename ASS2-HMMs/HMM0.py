import sys

def matrix_prod(mat1,mat2):

    res = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                res[i][j] += mat1[i][k] * mat2[k][j]

    return res

def load_matrix_line(l):
    M = []
    elems = l.strip().split(" ")
    n_row = int(elems[0])
    n_col = int(elems[1])
    numbers = elems[2:]

    for i in range(n_row):
        tmp = []
        for j in range(n_col):
            tmp.append(float(numbers[i*n_col+j]))
        M.append(tmp)

    return M

def format_output(m):
    rows = len(m)
    cols = len(m[0])

    out = str(rows) + " " + str(cols) + " "
    for i in range(len(m)):
        for j in range(len(m[0])):
            out += str(m[i][j]) + " "
        out += "\n"
    return out

lines = sys.stdin.readlines()

# transition matrix
A = load_matrix_line(lines[0])

# emission matrix
B = load_matrix_line(lines[1])

# initial probabilities
elems3 = lines[2].strip().split(" ")
p0 = [[float(i) for i in elems3[2:]]]

p1 = matrix_prod(p0, A)

observation_distribution = matrix_prod(p1, B)
out = format_output(observation_distribution)

print(out)