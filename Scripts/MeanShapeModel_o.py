import numpy as np
from scipy.spatial import distance

def XY_parameters_shape(shape, weight):
    X = np.dot(weight, np.transpose(shape[:, 0]))
    Y = np.dot(weight, np.transpose(shape[:, 1]))
    return X, Y

def compute_parameters(weight, reference_shape, shape):
    W = sum(weight)
    X1, Y1 = XY_parameters_shape(reference_shape, weight)
    X2, Y2 = XY_parameters_shape(shape, weight)
    Z = np.dot(weight, np.transpose(np.power(shape[:, 0], 2) + np.power(shape[:, 1], 2)))
    C1 = np.dot(weight, np.transpose(
        np.multiply(reference_shape[:, 0], shape[:, 0]) + np.multiply(reference_shape[:, 1], shape[:, 1])))
    C2 = np.dot(weight, np.transpose(
        np.multiply(reference_shape[:, 1], shape[:, 0]) - np.multiply(reference_shape[:, 0], shape[:, 1])))
    return W, X1, Y1, X2, Y2, Z, C1, C2

def solve_equations(weight, reference_shape, shape):
    W, X1, Y1, X2, Y2, Z, C1, C2 = compute_parameters(weight, reference_shape, shape)
    a = [[X2, -Y2, W, 0],
         [Y2, X2, 0, W],
         [Z, 0, X2, Y2],
         [0, Z, -Y2, X2]]
    b = [X1, Y1, C1, C2]
    x = np.linalg.solve(a, b)  # [ax, ay, tx, ty]
    return x

def map_shape(x, shape):
    X = ((x[0] * shape[:, 0]) - (x[1] * shape[:, 1])) + x[2]  # (ax*xk - ay*yk) + tx
    Y = ((x[1] * shape[:, 0]) + (x[0] * shape[:, 1])) + x[3]  # (ay*xk + ax*yk) + ty
    new_XY = np.transpose(np.concatenate([X[None, :], Y[None, :]]))
    return new_XY

def aligning_shapes(weight, data, current_mean):
    mean_s = []
    for i in range(24):
        tempx = []
        tempy = []
        for item in data:
            tempx.append(item[i, 0])
            tempy.append(item[i, 1])
        mean_s.append(np.mean(tempx))
        mean_s.append(np.mean(tempy))

    a = np.array(mean_s[::2])
    b = np.array(mean_s[1::2])
    new_mean = np.transpose(np.concatenate([a[None, :], b[None, :]]))

    # step 8 Normalizing mean
    x = solve_equations(weight, current_mean, new_mean)
    aligned_mean = map_shape(x, new_mean)
    # print(x)

    # step 9 Realigning data
    for i in range(len(data)):
        x = solve_equations(weight, aligned_mean, data[i])  # [ax, ay, tx, ty]
        aligned_data.append(map_shape(x, data[i]))

    return aligned_data, aligned_mean

def compute_weights(var_mat):
    tri = np.zeros((24, 24))
    tri[np.triu_indices(24, 1)] = var_mat
    var_sym = tri + np.transpose(tri)
    weight = []
    for i in range(24):
        weight.append(1.0 / sum(var_sym[i]))
    return weight

def var_points(dist_mat):
    var = []
    for i in range(len(dist_mat[0])):
        temp = []
        for item in dist_mat:
            temp.append(item[i])
        var.append(np.var(temp))
    return var

def shape_point_distance(x):
    dist = []
    for i in range(0, len(x)):
        dist.append(distance.pdist(x[i], metric='euclidean'))
    return dist

def extract_point_coord(train):
    fileNames = open("listmyfolder_o.txt", "r")
    files = fileNames.read().split("\n")
    XY = []
    files_to_train = []
    for item in train:
        files_to_train.append(files[item])
    for filename in files_to_train:
        t_file = open(filename, "r")
        lines = t_file.readlines()
        lines = [x.strip() for x in lines]
        lines.remove("version: 1")
        lines.remove("n_points: 24")
        lines.remove("{")
        lines.remove("}")
        try:
            L = []
            for i in lines:
                for j in i.split(" "):
                    L.append(float(j))

        except KeyError:
            print(filename)
            continue

        except ValueError:
            print(filename)
            continue

        XY.append(L)

    data_np = np.asarray(XY, dtype=float)
    data = []
    for i in range(0, len(XY)):
        a = np.transpose(data_np[i][::2])
        b = np.transpose(data_np[i][1::2])
        data.append(np.transpose(np.concatenate([a[None, :], b[None, :]])))

    return data

aligned_data = []

def main_mean(train):
    global aligned_data
    data = extract_point_coord(train)
    dist_matrix = shape_point_distance(data)        #step 1 R_kl
    var_mat = var_points(dist_matrix)               #step 2 V_R_kl
    weights = compute_weights(var_mat)              #step 3 w_k
    aligned_data.append(data[0])

    #step 5,6
    for i in range(1, len(data)):
        x = solve_equations(weights, data[0], data[i])  # [ax, ay, tx, ty]
        aligned_data.append(map_shape(x, data[i]))

    #step 7 computing mean
    mean_s = []
    for i in range(24):
        tempx = []
        tempy = []
        for item in aligned_data:
            tempx.append(item[i, 0])
            tempy.append(item[i, 1])
        mean_s.append(np.mean(tempx))
        mean_s.append(np.mean(tempy))

    a = np.array(mean_s[::2])
    b = np.array(mean_s[1::2])
    mean_shape = np.transpose(np.concatenate([a[None, :], b[None, :]]))

    #step 8 Normalizing mean
    x = solve_equations(weights, data[0], mean_shape)
    aligned_mean = map_shape(x, mean_shape)

    #step 9 Realigning data
    for i in range(0, len(data)):
        x = solve_equations(weights, aligned_mean, data[i])  # [ax, ay, tx, ty]
        aligned_data.append(map_shape(x, data[i]))

    #step 10 Till Convergence
    new_mean = np.zeros((24,2))
    current_mean = aligned_mean
    test_x = sum(current_mean[:, 0] - new_mean[:, 0])
    test_y = sum(current_mean[:, 1] - new_mean[:, 1])
    no_steps = 0
    diff = 10

    while diff >= 0.03:
        aligned_data, new_mean = aligning_shapes(weights, aligned_data, aligned_mean)
        test_x_new = sum(current_mean[:, 0] - new_mean[:, 0])
        test_y_new = sum(current_mean[:, 1] - new_mean[:, 1])
        current_mean = new_mean
        diff = abs(test_x - test_x_new) + abs(test_y - test_y_new)
        test_x = test_x_new
        test_y = test_y_new
        no_steps += 1
        #print("iter: " + str(no_steps) + " diff: " + str(diff))

    #Writing to file
    mean_shape_file = open("meanShape_o.pts", "w")
    mean_shape_file.writelines("version: 1\nn_points: 24\n{\n")
    for i in range(24):
        mean_shape_file.writelines("%f " %j for j in [current_mean[i, 0], current_mean[i, 1]])
        mean_shape_file.writelines("\n")
    mean_shape_file.writelines("}")

    #Writing aligned data to file
    al_points = []
    for item in data:
        temp = []
        for i in range(24):
            temp.append(item[i, 0])
            temp.append(item[i, 1])
        al_points.append(temp)

    aligned_data_file = open("alignedData_o.txt", "w")
    for i in range(len(al_points)):
        aligned_data_file.writelines("%f " %j for j in al_points[i])
        if i != (len(al_points) - 1):
            aligned_data_file.writelines("\n")
    return al_points

#main_mean(range(407))