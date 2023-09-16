import sys

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

alpha = [[0 for _ in range(N)] for _ in range(T)]

for i in range(N):
    alpha[0][i] = p0[i] * B[i][obs[0]]

for t in range(1, T):
    for i in range(N):
        for j in range(N):
            alpha[t][i] += alpha[t - 1][j] * A[j][i] * B[i][obs[t]]

out = sum(alpha[:][T-1])

print(out)




