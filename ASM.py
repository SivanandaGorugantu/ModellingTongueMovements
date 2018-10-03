import numpy as np
from scipy import spatial
from scipy.optimize import fsolve
from scipy.ndimage import filters
import cv2
import math
import MeanShapeModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

def orderingShape(x):
    x_axis = np.array(x[::2])
    y_axis = np.array(x[1::2])
    dist = []
    u = []
    u.append(x_axis[0])
    u.append(y_axis[0])
    for i in range(len(x_axis)):
        v = []
        v.append(x_axis[i])
        v.append(y_axis[i])
        dist.append(spatial.distance.euclidean(u, v))

    sort_index = np.argsort(dist)
    x_axis_sorted = x_axis[sort_index]
    y_axis_sorted = y_axis[sort_index]

    x_axis_new = []
    y_axis_new = []
    for i in x_axis_sorted[::2]:
        x_axis_new.append(i)
    for i in x_axis_sorted[-1::-2]:
        x_axis_new.append(i)
    for i in y_axis_sorted[::2]:
        y_axis_new.append(i)
    for i in y_axis_sorted[-1::-2]:
        y_axis_new.append(i)

    shape = np.transpose(np.concatenate([np.array(x_axis_new)[None, :], np.array(y_axis_new)[None, :]]))
    o_shape = []
    for i in range(len(x_axis_new)):
        o_shape.append(x_axis_new[i])
        o_shape.append(y_axis_new[i])

    return x_axis_new, y_axis_new, shape, o_shape

def obtain_mean(data):
    '''data_file = open("alignedData.txt", "r")
    content = data_file.readlines()
    content = [x.strip() for x in content]
    data = []
    for i in content:
        temp = []
        for j in i.split(" "):
            temp.append(float(j))
        data.append(temp)
    '''
    ordered_data = []
    ordered_shape = []
    for item in data:
        x, y, temp_shape, temp_ordered = orderingShape(item)
        ordered_data.append(temp_shape)
        ordered_shape.append(temp_ordered)

    N = len(ordered_data)
    x_mean_x = []
    x_mean_y = []
    for i in range(len(ordered_data[0])):
        temp_x = []
        temp_y = []
        for item in ordered_data:
            temp_x.append(item[i, 0])
            temp_y.append(item[i, 1])
        x_mean_x.append((1/N) * sum(temp_x))
        x_mean_y.append((1 / N) * sum(temp_y))

    x_axis = x_mean_x
    y_axis = x_mean_y
    weight = compute_weights(ordered_data)
    shape = ret_shape(x_axis, y_axis)
    return x_axis, y_axis, shape, weight, ordered_shape

def eig_vect_P(data):
    cov = np.cov(data, rowvar=False)
    e_val, P = np.linalg.eig(cov)  # e_val - eigenvalues and P - eigenvectors

    sort_index = np.argsort(-1 * e_val)
    e_val = e_val[sort_index]
    P = P[:, sort_index]
    pov = 0
    t = 0

    while pov < 0.95:
        t = t + 1
        pov = sum(e_val[0:t]) / sum(e_val)
        #print("t: " + str(t) + " pov: " + str(pov))

    e_val_t = e_val[0:t]
    P_t = P[:, 0:t]
    C_inv = np.linalg.inv(cov)
    return t, e_val_t, P_t, C_inv

def ret_shape(x_axis, y_axis):
    shape = np.transpose(np.concatenate([np.array(x_axis)[None, :], np.array(y_axis)[None, :]]))
    return shape

def ret_xy(shape):
    x_param = shape[:, 0]
    y_param = shape[:, 1]
    return x_param, y_param

def compute_weights(x):
    var_mat = var_points(x)
    tri = np.zeros((24, 24))
    tri[np.triu_indices(24, 1)] = var_mat
    var_sym = tri + np.transpose(tri)
    weight = []
    for i in range(24):
        weight.append(1.0 / sum(var_sym[i]))
    return weight

def var_points(x):
    dist_mat = shape_point_distance(x)
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
        dist.append(spatial.distance.pdist(x[i], metric='euclidean'))
    return dist

def centroid(x_param, y_param):
    X_centroid = np.mean(x_param)
    Y_centroid = np.mean(y_param)
    return X_centroid, Y_centroid

def strong_edge_search(x_local, y_local, I):
    #pyplot.ion()
    #fig = pyplot.figure()
    #ax = fig.add_subplot(1, 1, 1)
    x_local.append(x_local[0])
    y_local.append(y_local[0])
    dX = []
    dY = []
    for i in range(24):
        slope = (y_local[i + 1] - y_local[i]) / (x_local[i + 1] - x_local[i])
        try:
            normal_slope = -1 / slope
        except ZeroDivisionError:
            normal_slope = -1 / 0.000001
        angle = np.arctan(slope)
        normal_angle = angle + math.pi / 2
        if np.rad2deg(normal_angle) >= 360:
            normal_angle = angle - math.pi / 2
        intercept = y_local[i + 1] - (x_local[i + 1] * normal_slope)
        x_range = range(600)
        normal_x = x_range[int(x_local[i + 1]) - 10: int(x_local[i + 1]) + 10]
        normal_y = np.multiply(normal_slope, normal_x) + intercept
        normal_y = np.array(normal_y)
        if 80 <= np.rad2deg(normal_angle) <= 100:
            normal_x = [int(x_local[i + 1])] * len(normal_x)
            normal_y = range(int(y_local[i + 1] - 10), int(y_local[i + 1] + 10))
        normal_x = np.around(normal_x)
        normal_y = np.around(normal_y)
        temp = []
        for k in range(len(normal_x)):
            try:
                temp.append(I[int(normal_x[k]), int(normal_y[k]), 0])
            except IndexError:
                print("appending normal: IndexError")
                print(normal_x[k], normal_y[k])

        #Finding Edges
        if i > 13:
            mask = [-30, 0, 30]
        else:
            mask = [30, 0, -30]
        edge = []
        for l in range(1, len(temp) - 1):
            t = np.multiply([temp[l-1], temp[l], temp[l+1]], mask)
            edge.append(abs(sum(t)))
        #ax.clear()
        #ax.plot(edge)
        #print("Edge for point: " + str(i) + str(edge))
        #print("slope: " + str(np.rad2deg(normal_angle)))
        ind = np.nanargmax(edge)
        #ax.plot(edge[ind], 'ro')
        #pyplot.pause(2)
        #fig.canvas.draw()
        if edge[int(ind)] == 0:
            dX.append(int(x_local[i + 1]))
            dY.append(int(y_local[i + 1]))
        else:
            dX.append(normal_x[int(ind + 1)])
            dY.append(normal_y[ind + 1])
    x_local.pop()
    y_local.pop()
    return dX, dY

def compute_dX(x_param, y_param, x_new, y_new):
    temp_x = []
    temp_y = []
    for i in range(len(x_param)):
        tx = x_param[i] - x_new[i]
        ty = y_param[i] - y_new[i]
        temp_x.append(tx)
        temp_y.append(ty)
    dX = ret_shape(temp_x, temp_y)
    return dX

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
    #print("W: "+ str(W) + "\nX1: "+ str(X1)+ "\nX2: "+ str(X2) + "\nY1: "+ str(Y1) +"\nY2: "+ str(Y2) +"\nZ: "+ str(Z) +"\nC1: "+ str(C1) + "\nC2: "+ str(C2))
    a = [[X2, -Y2, W, 0],
         [Y2, X2, 0, W],
         [Z, 0, X2, Y2],
         [0, Z, -Y2, X2]]
    b = [X1, Y1, C1, C2]
    x = np.linalg.solve(a, b)  # [ax, ay, tx, ty]
    return x

def map_shape(weight, reference_shape, shape):
    x = solve_equations(weight, reference_shape, shape)
    X = ((x[0] * shape[:, 0]) - (x[1] * shape[:, 1])) + x[2]  # (ax*xk - ay*yk) + tx
    Y = ((x[1] * shape[:, 0]) + (x[0] * shape[:, 1])) + x[3]  # (ay*xk + ax*yk) + ty
    new_XY = ret_shape(X, Y)
    return x, new_XY

def compute_pose(pose_param=[0, 0, 0, 0]):
    def equation(p):
        r, q = p
        f1 = r * math.cos(math.radians(q)) - pose_param[0]
        f2 = r * math.sin(math.radians(q)) - pose_param[1]
        return (f1, f2)

    s, theta = fsolve(equation, (0, 0))
    return s, theta

def find_dx(s, alpha, theta, d_theta, x, dX, dXc, dYc):
    ax = s * math.cos(theta)
    ay = s * math.sin(theta)
    X = ((ax * x[:, 0]) - (ay * x[:, 1]))
    Y = ((ay * x[:, 0]) + (ax * x[:, 1]))
    temp_x = X + dX[:, 0] - dXc
    temp_y = Y + dX[:, 1] - dYc
    y = ret_shape(temp_x, temp_y)
    ax = (1 / (s * (1 + alpha))) * math.cos(-(theta + d_theta))
    ax = (1 / (s * (1 + alpha))) * math.sin(-(theta + d_theta))
    X = ((ax * y[:, 0]) - (ay * y[:, 1]))
    Y = ((ay * y[:, 0]) + (ax * y[:, 1]))
    temp_x = X - x[:, 0]
    temp_y = Y - x[:, 1]
    dx = ret_shape(temp_x, temp_y)
    return dx

def order_points(o_dx):
    o_points = []
    for i in range(len(o_dx[:, 0])):
        o_points.append(o_dx[i, 0])
        o_points.append(o_dx[i, 1])
    return o_points

def compute_db(o_dx, P):
    dx = order_points(o_dx)
    db = np.matmul(np.transpose(P), dx)
    return db

def split_points(shape):
    x_params = shape[::2]
    y_params = shape[1::2]
    new_shape = ret_shape(x_params, y_params)
    return new_shape

def compute_Pb(P, b, db):
    Wb = np.identity(16)
    #temp = b + np.matmul(Wb, db)
    temp = b + db
    temp_Pb = np.matmul(P, temp)
    Pb = split_points(temp_Pb)
    return Pb, temp

def draw_contour(image, x_local, y_local):
    x_local.append(x_local[0])
    y_local.append(y_local[0])
    points = np.around(ret_shape(x_local, y_local)).astype(np.int32)
    contour = []
    for i in range(len(points)):
        contour.append([points[i]])
    contour = np.array(contour)
    #print("contour: " + str(contour))
    cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
    x_local.pop()
    y_local.pop()

def show_contour(img, x, y, delay):
    cv2.imshow("Original", img)
    draw_contour(img, x, y)
    cv2.imshow("contour", img)
    cv2.waitKey(delay)

def compute_Wb(data, x_mean, P, t):
    W = np.identity(t)
    t_space_data = []
    for i in range(len(data)):
        t = np.array(data[i]) - np.array(x_mean)
        temp = np.matmul(np.transpose(P), t)
        t_space_data.append(temp)
    std_dev_vals = np.std(t_space_data, axis=0)
    Wb = np.matmul(W, std_dev_vals)
    Wb = np.divide((1 / len(data)), Wb)
    return Wb

def measure_error(gold_true, shape_pred):
    e = math.sqrt(mean_squared_error(gold_true, shape_pred))
    return e

def figure_of_merit(gold_x, obtained_x, gold_y, obtained_y):
    dist = 0
    for i in range(24):
        dist += spatial.distance.euclidean([gold_x[i], gold_y[i]], [obtained_x[i], obtained_y[i]])
    return (dist/len(gold_x))

d = {}
with open("allFilesDictionary.txt") as f:
    for line in f:
       (key, val) = line.split()
       d[key] = val


train_fileNames = open("FileNames.txt", "r")
train_files = train_fileNames.read().split("\n")
kf = KFold(n_splits=10)
folds = []
fold_error = []
for item in kf.split(train_files):
    folds.append(item)
for iter in range(9,10):
    print("Building model for fold: " + str(iter))
    dat = MeanShapeModel.main_mean(list(folds[iter])[0])
    print("Model built for fold: " + str(iter))
    x, y, mean_shape, weights, ordered_data = obtain_mean(dat)
    mean_x = x
    mean_y = y
    t, e_val, P, C_inv = eig_vect_P(ordered_data)
    x_bar = order_points(mean_shape)
    Wb = compute_Wb(ordered_data, x_bar, P, t)
    x_mean = np.mean(ordered_data, axis=0)
    D = []
    for i in range(len(ordered_data)):
        D.append(spatial.distance.mahalanobis(x_mean, ordered_data[i], C_inv))
    D_max = max(D)
    Xc, Yc = centroid(x, y)
    temp_shape = ret_shape(([Xc] * len(x)), ([Yc] * len(y)))
    try:
        pose_params, temp_shape = map_shape(weights, mean_shape, temp_shape)
        s, theta = compute_pose(pose_params)
    except:
        print("Linalg Error: Singular Matrix \nSetting (s,theta) to (1, 90)")
        s = 1
        theta = 90
    wt = len(ordered_data) / (600 * 480)
    w_theta = theta / math.pi
    ws = 1
    run = 0
    diff = 10
    #I = cv2.imread("PS__060418143403_59.jpg")       #, cv2.IMREAD_GRAYSCALE)
    frame = 0
    i_error = []
    count = 0
    for ls in list(folds[iter])[1]:
        val_file = train_files[ls]
        try:
            key_file = [ke for ke, value in d.items() if value == val_file][0]
            filename = d[key_file]
        except IndexError:
            count += 1
            continue
        x, y, mean_shape, weights, ordered_data = obtain_mean(dat)
        frame += 1
        print("Frame: " + str(frame))
        Xc, Yc = centroid(x, y)
        temp_shape = ret_shape(([Xc] * len(x)), ([Yc] * len(y)))
        try:
            pose_params, temp_shape = map_shape(weights, mean_shape, temp_shape)
            s, theta = compute_pose(pose_params)
        except:
            #print("Linalg Error: Singular Matrix \nSetting (s,theta) to (1, 90)")
            s = 1
            theta = 90
        wt = len(ordered_data) / (600 * 480)
        w_theta = theta / math.pi
        ws = 1
        run = 0
        diff = 10
        I = cv2.imread(str(key_file))
        #I_blur = cv2.GaussianBlur(I, (3, 3), sigmaX=3, sigmaY=3)
        I_blur = filters.rank_filter(I, -1, (3, 3, 2))
        #while I_diff > 0.5:
        #   I_blur = cv2.medianBlur(I_blur, 3)
        while (abs(diff)) > 0.1:
                x_shifted, y_shifted = strong_edge_search(x, y, I_blur)
                dX = compute_dX(x, y, x_shifted, y_shifted)
                pose_params, mapped_mean_shape = map_shape(weights, (mean_shape + dX), mean_shape)
                alpha, d_theta = compute_pose(pose_params)          #alpha = (1 + ds)
                dXc = pose_params[2]
                dYc = pose_params[3]
                dx = find_dx(s, alpha, theta, d_theta, mean_shape, dX, dXc, dYc)    # dx in model frame
                b = np.zeros(t)
                db = compute_db(dx, P)
                Pb, b = compute_Pb(P, b, db)
                new_x = (mean_shape + Pb)
                Dm = sum((db ** 2) / e_val)

                if Dm > D_max:
                    #print("Dmax = " + str(D_max) + " and Dm = " + str(Dm))
                    #for m in range(t):
                        #b[m] = (b[m]) * e_val[m] / (Dm)
                    for m in range(t):
                        if not (-1 * math.sqrt(e_val[m])) < b[m] < (1 * math.sqrt(e_val[m])):       # and m not in [0, 1, 2, 3, 4, 5]:     # or m != 3:
                            #b[m] = math.sqrt(e_val[m]) / b[m]
                            b[m] = b[m] * (D_max / Dm)
                    Pb, b = compute_Pb(P, b, np.zeros(t))
                    new_x = (mean_shape + Pb)

                    #print("changed new_x")
                    #diff = D_max / Dm
                diff = sum((mean_shape[:, 0] - new_x[:, 0]) + (mean_shape[:, 1] - new_x[:, 1]))
                Xc += wt * dXc
                Yc += wt * dYc
                theta += w_theta * d_theta
                s = s * (1 + (ws * alpha))
                mean_shape = new_x
                x = mean_shape[:, 0]
                y = mean_shape[:, 1]
                x = np.ndarray.tolist(x)
                y = np.ndarray.tolist(y)
                run += 1
                #print("run: " + str(run) + " diff: " + str(diff))
        show_contour(I, x, y, 300)
        '''try:
            with open(d[filename], "r") as f:
                lines = f.readlines()
                lines = [tem.strip() for tem in lines]
                lines.remove("version: 1")
                lines.remove("n_points: 24")
                lines.remove("{")
                lines.remove("}")
                L = []
                for item in lines:
                    for j in item.split(" "):
                        L.append(float(j))
                gold_x, gold_y, gold_shape, gold_o_shape = orderingShape(L)
                fom = figure_of_merit(gold_x, x, gold_y, y)
                d_file = open("figureMerit_gold.txt", "a")
                d_file.writelines("%f" %fom)
                d_file.writelines("\n")
                fom = figure_of_merit(mean_x, x, mean_y, y)
                m_file = open("figureMerit_mean.txt", "a")
                m_file.writelines("%f" % fom)
                m_file.writelines("\n")
        except KeyError:
            continue
    '''
    '''   #try:
        with open(val_file, "r") as f:
            lines = f.readlines()
            lines = [tem.strip() for tem in lines]
            lines.remove("version: 1")
            lines.remove("n_points: 24")
            lines.remove("{")
            lines.remove("}")
            L = []
            for item in lines:
                for j in item.split(" "):
                    L.append(float(j))
            gold_x, gold_y, gold_shape, gold_o_shape = orderingShape(L)
            error = measure_error(gold_shape, mean_shape)
            #print(error)
            i_error.append(error)
            d_file = open("error_fold"+str(iter)+".txt", "a")
            d_file.writelines("%f" %error)
            d_file.writelines("\n")
    fold_error.append(np.mean(i_error))
    print("Missed " + str(count) + " files.")'''
    #print("Error of fold " + str(iter) + ": " + str(fold_error[iter]))
'''total_error = np.mean(fold_error)
print("Total Error: " + str(total_error))
stat = open("stats.txt", "w")
for it in fold_error:
    stat.writelines("%f" %it)
    stat.writelines("\n")
stat.writelines("cross validation error: %f" %total_error)'''