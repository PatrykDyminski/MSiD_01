# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x. Elo
    """

    '''
    matrix = design_matrix(x, w.size-1)
    y_d = matrix @ w
    sub = y - y_d
    mult = np.linalg.norm(sub,2)**2
    return mult/x.size
    '''

    sub = y - polynomial(x, w)
    sum = np.sum(sub**2)
    return sum / x.size



def design_matrix(x_train, M):
    """
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    """
    '''
    matrix = np.empty((x_train.size, M+1))

    for i in range(x_train.size):
        for j in range(M+1):
            matrix[i][j] = x_train[i]**j
    '''

    return np.array([x_train[:, 0] ** i for i in range(M+1)]).transpose()


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    """

    dm = design_matrix(x_train, M)
    dm_t = dm.transpose()

    w = np.linalg.inv(dm_t @ dm) @ dm_t @ y_train
    err = mean_squared_error(x_train, y_train, w)

    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    """

    dm = design_matrix(x_train, M)
    dm_t = dm.transpose()

    w = np.linalg.inv((dm_t @ dm) + np.identity(M+1) * regularization_lambda) @ dm_t @ y_train
    err = mean_squared_error(x_train, y_train, w)

    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    """

    tab = np.array([least_squares(x_train, y_train, m) for m in M_values])
    m_errors = np.array([mean_squared_error(x_val, y_val, w) for (w, err) in tab])

    index = m_errors.argmin().item(0)

    return tab[index][0], tab[index][1], m_errors[index]


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    """

    tab = np.array([regularized_least_squares(x_train, y_train, M, lam) for lam in lambda_values])
    m_errors = np.array([mean_squared_error(x_val, y_val, w) for (w, err) in tab])

    index = m_errors.argmin().item(0)

    return tab[index][0], tab[index][1], m_errors[index], lambda_values[index]
