import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_squared_error, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
import seaborn as sns
from scipy import stats


color_palette_divergent = LinearSegmentedColormap.from_list('ee', ['#E27872', '#F9F9F8', '#509A9A'])
color_palette_discrete = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
color_palette_sequential = [ '#ece3f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636']

markers = ['o', '^', '*','H', 'P', 'D', 'X', 'h', 'p', 'd', 'c']

def plot_class_report(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_class_report.

    :y_test: TODO
    :y_Hat: TODO
    :classes_labels: TODO
    :returns: TODO

    """
    tmp_report = classification_report(y_test, y_hat, output_dict=True)
    targets = list(classes_labels)
    targets.append('average')
    tmp_report = pd.DataFrame(tmp_report)\
                    .drop(columns=['weighted avg', 'macro avg'])
    tmp_report.columns = targets
    tmp_report = tmp_report.drop(labels='support')
    tmp_report = tmp_report.drop(columns='average')
    tmp_report = tmp_report.T

    for index, (colname, serie) in enumerate(tmp_report.iteritems()):
        plt.subplot(3, 1, index + 1)
        serie.plot(kind = 'barh')
        plt.title(f"Métrica: {colname}")
        plt.tight_layout()

def plot_confusion_matrix(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_confusion_matrix.

    :y_test: TODO
    :y_hat: TODO
    :returns: TODO

    """
    tmp_confused = confusion_matrix(y_test, y_hat)
    custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)
    sns.heatmap(tmp_confused, annot=True, cbar=False, cmap=custom_cmap, xticklabels=classes_labels,
                yticklabels=classes_labels)
    plt.xlabel('Classes on testing data')
    plt.ylabel('Predicted classes on training')
    plt.grid(False)

def probability_contours(model,df,target, x1, x2,classes_labels, fill_contours=False):
    """TODO: Docstring for probability_contours.

    :model: TODO
    :df: TODO
    :target: TODO
    :x1: TODO
    :x2: TODO
    :fill_contours: TODO
    :returns: TODO

    """
    # a partir de dos columnas separadas
    tmp_x = df.loc[:, [x1, x2]]
    # estimar un modelo en base a las columnas y el vector objetivo
    tmp_model = model.fit(tmp_x, target)
    # extraemos los puntos ij
    x_0, x_1 = generate_mesh_grid(tmp_x, x1, x2)
    # Aplanamos cada punto ij, y los concatenamos
    map_x = np.c_[x_0.ravel(), x_1.ravel()]
    # implementamos la predicción de Pr(y)
    predict_y_pr = tmp_model.predict_proba(map_x)
    # implementamos la predicción de la clase con argmax
    predict_y = tmp_model.predict(map_x)
    # extraemos los límites de probabilidad
    boundaries_pr = predict_y_pr[:, 1].reshape(x_1.shape)
    # extraemos las clases
    boundaries_y = predict_y.reshape(x_0.shape)
    custom_cmap = ListedColormap(color_palette_sequential)

    # por cada clase estimable
    for i in target.unique():
        # graficamos los puntos correspondientes en las dos columnas
        plt.plot(tmp_x[target == i][x1], tmp_x[target == i][x2],
                 '.', marker=markers[i], color=color_palette_discrete[i],
                 label = "{}".format(classes_labels[i]), alpha=.8)
    # Graficamos los límites
    if fill_contours is True:
        custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)
        plt.contourf(x_0, x_1, boundaries_pr, cmap=custom_cmap)
        plt.colorbar()
        plt.clim(0, 1)
    else:
        vis_boundaries = plt.contour(x_0, x_1, boundaries_pr, cmap = custom_cmap)
        plt.clabel(vis_boundaries, inline=1)

    plt.legend(framealpha=0.5, edgecolor='slategrey', fancybox=True)
    plt.xlabel(x1)
    plt.ylabel(x2)

def generate_mesh_grid(df, x1, x2):
    """TODO: Docstring for generate_mesh_grid.

    :df: TODO
    :x1: TODO
    :x2: TODO
    :returns: TODO

    """

    # a partir de dos columnas separadas
    tmp_x = df.loc[:, [x1, x2]]
    # retornar una red con puntos ij
    x_0, x_1 = np.meshgrid(
        # considerando el mínimo y máximo de x1, simulado 100 y reescalado entre -1 y 1
        np.linspace(np.min(tmp_x[x1]), np.max(tmp_x[x1]), num=100).reshape(-1, 1),
        # considerando el mínimo y máximo de x2, simulado 100 y reescalado entre -1 y 1
        np.linspace(np.min(tmp_x[x2]), np.max(tmp_x[x2]), num=100).reshape(-1, 1)
    )
    return x_0, x_1

def compare_priors(X_train, X_test, y_train, y_test, prior):
    """TODO: Docstring for compare_priors.

    :prior: TODO
    :returns: TODO

    """
    tmp_clf = BernoulliNB(class_prior=prior)
    tmp_clf.fit(X_train, y_train)
    tmp_class = tmp_clf.predict(X_test)
    tmp_pr = tmp_clf.predict_proba(X_test)[:, 1]
    tmp_acc = accuracy_score(y_test, tmp_class).round(3)
    tmp_rec = recall_score(y_test, tmp_class).round(3)
    tmp_prec = precision_score(y_test, tmp_class).round(3)
    tmp_f1 = f1_score(y_test, tmp_class).round(3)
    tmp_auc = roc_auc_score(y_test, tmp_pr).round(3)
    print("A priori: {0}\nAccuracy: {1}\nRecall: {2}\nPrecision: {3}\nF1: {4}\nAUC: {5}\n".format(prior, tmp_acc, tmp_rec, tmp_prec, tmp_f1, tmp_auc))