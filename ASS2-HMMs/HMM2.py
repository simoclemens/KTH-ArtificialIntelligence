import sys


def vector_prod(mat1, mat2):

    res = 0

    for i in range(len(mat1)):
        res += mat1[i] * mat2[i]

    return res


# lines = open(sys.argv[1]).readlines()
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

# emission matrix
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
p0 = [float(i) for i in elems3[2:]]

# observations sequence
elems4 = lines[3].strip().split(" ")
obs = [int(i) for i in elems4[1:]]

T = len(obs)
N = len(A)

delta = [[0 for _ in range(N)] for _ in range(T)]
delta_idx = [[0 for _ in range(N)] for _ in range(T)]

for i in range(N):
    delta[0][i] = p0[i] * B[i][obs[0]]

for t in range(1, T):
    for i in range(N):
        tmp = []
        for j in range(N):
            tmp.append(A[j][i] * delta[t-1][j] * B[i][obs[t]])
        delta[t][i] = max(tmp)
        delta_idx[t][i] = tmp.index(max(tmp))

state_est = [0 for _ in range(T)]

state_est[T-1] = delta[T-1].index(max(delta[T-1]))

for t in range(T-2,-1,-1):
    state_est[t] = int(delta_idx[t+1][state_est[t+1]])

for i in range(T):
    print(state_est[i], end=" ")








