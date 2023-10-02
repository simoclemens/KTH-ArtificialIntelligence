import sys
import math

lines = open(sys.argv[1]).readlines()
#lines = sys.stdin.readlines()

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
K = len(B[0])
MAX_ITER = 50

mle = float("-inf")
mle_new = float("-inf")
iteration = 0

# while (mle_new > mle or mle_new == float("-inf")) and iteration <= MAX_ITER:
while iteration <= MAX_ITER:
    print(mle_new)
    mle = mle_new
    iteration += 1

    # ALPHA
    # initialization
    alpha = [[0 for _ in range(N)] for _ in range(T)]
    c_log = []
    c = []

    # first row
    for i in range(N):
        alpha[0][i] = p0[i] * B[i][obs[0]]

    c_t = 1 / sum(alpha[0])

    for i in range(N):
        alpha[0][i] *= c_t

    c.append(c_t)
    c_log.append(math.log(c_t))

    # computation
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha[t][i] += alpha[t - 1][j] * A[j][i]
            alpha[t][i] *= B[i][obs[t]]

        c_t = 1 / sum(alpha[t])

        for i in range(N):
            alpha[t][i] *= c_t
        c.append(c_t)
        c_log.append(math.log(c_t))

    mle_new = -sum(c_log)

    # BETA
    # initialization
    beta = [[0 for _ in range(N)] for _ in range(T)]

    # first row
    for i in range(N):
        beta[T-1][i] = c[T-1]

    # computation
    for t in range(T-2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[t][i] += beta[t+1][j] * B[j][obs[t+1]] * A[i][j]
            beta[t][i] *= c[t]

    # DI-GAMMA/GAMMA

    # initialization
    di_gamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T)]
    gamma = [[0 for _ in range(N)] for _ in range(T)]

    # computation
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                di_gamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j]
            gamma[t][i] = sum(di_gamma[t][i])

    # last row
    gamma[T-1] = alpha[T-1]

    # initial state update
    p0 = gamma[0]

    # A update
    for i in range(N):
        den = 0
        for t in range(T-1):
            den += gamma[t][i]
        for j in range(N):
            num = 0
            for t in range(T-1):
                num += di_gamma[t][i][j]
            A[i][j] = num / den

    # B update
    for i in range(N):
        for k in range(K):
            den = 0
            num = 0
            for t in range(T):
                den += gamma[t][i]
                if obs[t] == k:
                    num += gamma[t][i]
            B[i][k] = num / den



out1 = ""
out1 += str(N) + " " + str(N)
for i in range(N):
    for j in range(N):
        out1 += " " + str(A[i][j])
out1 += "\n"

out2 = ""
out1 += str(N) + " " + str(K)
for i in range(N):
    for j in range(K):
        out1 += " " + str(B[i][j])

print(out1)
print(out2)