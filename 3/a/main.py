import numpy as np
from matplotlib import pyplot
from surprise import Dataset
from surprise import KNNBasic
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


def get_user_predictions(predictions, raw_user_id):
    """
    Filters all predictions in order to retain only the ones of the specified user.

    :param predictions: All predictions determined after running an algorithm like KNN or SVD on a given dataset.
    :param raw_user_id: The raw ID of the targeted user (read this: https://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note).
    :return: The predictions of the specified user, or an empty list if the ID is unknown.
    """
    user_predictions = []
    for prediction in predictions:
        if prediction.uid == raw_user_id:
            user_predictions.append(prediction)
    return user_predictions


def get_prediction_estimation(prediction):
    """
    Gets the estimation value of a given prediction (function used for sort purpose).

    :param prediction: The prediction to convert into the estimation rating.
    :return: The estimation value of a given prediction.
    """
    return prediction.est


def get_top_n_predictions(u_predictions, n):
    """
    Determines the top N predictions of user's predictions sorted from the the highest estimate value in first position
    to the lowest one in last position.

    :param u_predictions: The user's predictions.
    :param n: The top N predictions to return.
    :return: The top N predictions.
    """
    top_predictions = u_predictions.copy()
    top_predictions.sort(key=get_prediction_estimation, reverse=True)
    return top_predictions[0:n]


def get_nb_true_positives(top_n_predictions):
    """
    Determines and returns the number of true positives in the predictions.
    A true positive correspond to a recommended item which is embedded in the relevant elements for the
    corresponding user. In easier words: in reality the item match with preferences of the user, so the recommendation
    is interesting for him.

    :param top_n_predictions: The top N predictions where the estimated rating is the highest.
    :return: The number of true positives in the predictions.
    """
    nb_true_positives = 0
    for prediction in top_n_predictions:
        if prediction.r_ui >= 4:
            nb_true_positives = nb_true_positives + 1
    return nb_true_positives


def get_nb_relevant_elements(user_predictions):
    """
    Gets the number of relevant elements among all not-rating yet items.
    We expect here that relevant elements are those rated with 4 or 5 stars by the user (r_ui field).

    :param user_predictions: The predictions for a user.
    :return: The number of relevant elements for the user.
    """
    nb_relevant_elements = 0
    for user_prediction in user_predictions:
        if user_prediction.r_ui >= 4:
            nb_relevant_elements = nb_relevant_elements + 1
    return nb_relevant_elements


def get_precision(top_n_predictions):
    """
    Calculates and returns the precision of the predictions.

    :param top_n_predictions: The top N predictions where the estimated rating is the highest.
    :return: The precision of the predictions.
    """
    return get_nb_true_positives(top_n_predictions) / len(top_n_predictions)


def get_recall(top_n_predictions, user_predictions):
    """
    Calculates and returns the recall of the predictions.

    :param top_n_predictions: The top N predictions where the estimated rating is the highest.
    :param user_predictions: The predictions for a user.
    :return: The recall of the predictions.
    """
    return get_nb_true_positives(top_n_predictions) / get_nb_relevant_elements(user_predictions)


def get_f1(precision, recall):
    """
    Gets F1 measure for evaluating predictions by combining precision and recall ones together.

    :param precision: The precision of the predictions.
    :param recall: The recall of the predictions.
    :return: The F1 measure according to the specified precision and recall values.
    """
    return (2 * precision * recall) / (precision + recall)


def print_recommendations(top_predictions):
    """
    Prints recommendations for the user associated this specified top predictions. It is assume here that all
    predictions of array target the same user.

    :param top_predictions: The top predictions to display. The better prediction is assumed to be at
        the first position in the array.
    """
    if len(top_predictions) > 0:
        i = 1
        raw_user_id = top_predictions[0].uid
        print("Top {} recommended-items for the user {}".format(len(top_predictions), raw_user_id))
        for predict in top_predictions:
            print("#{}: item {} estimated {:.2f}/5, reality {}/5".format(i, predict.iid, predict.est, predict.r_ui))
            i = i + 1


def print_recommendations_evaluation(precision, recall, f1):
    """
    Prints evaluation of the recommendations.

    :param precision: The precision of the recommendations.
    :param recall: The recall of the recommendations.
    :param f1: The F1 measure of the recommendations.
    """
    print("Evaluation of recommendations")
    print("Precision =", round(precision, 3))
    print("Recall =", round(recall, 3))
    print("F1 =", round(f1, 3))


def plot_maes(k_list, maes):
    """
    Shows the curve plot which correspond the given axes.
    The X-axis is associated to the arg "maes".
    The Y-axis is associated to the arg "k_list".

    :param k_list: The list of K where MAEs values are associated with them.
    :param maes: The MAEs values of the corresponding K values.
    """
    pyplot.plot(k_list, maes)
    pyplot.title('MAE evolution depending on K value')
    pyplot.xlabel('K nearest-neighbors')
    pyplot.ylabel('MAE')
    pyplot.show()


def dichotomy_search_k(data, sim_options_knn):
    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=.25)

    # Adjust range values according to the experience to realize
    k_list = list(range(40, 100, 5))

    # Store MAEs values to display them later
    maes = []

    for k in k_list:
        algo = KNNBasic(k=k, sim_options=sim_options_knn)
        predictions = algo.fit(train_set).test(test_set)
        maes.append(mae(predictions))

    plot_maes(k_list, maes)


def probabilistic_analysis_search_k(data, sim_options_knn):
    # Number of experiments (>= 30 to permit quantitative analysis)
    n = 1

    # We focus on the range of values of K where MAEs remain unstable
    k_list = list(range(40, 80, 4))

    # Matrix of MAEs values: row correspond to an experiment and column to a K value
    # A way to read a cell: for the experiment n°X with K=Y the MAE value is equal to Z (the value of the cell)
    m_maes = np.zeros((n, len(k_list)))

    for i in range(0, n, 1):
        # Split dataset in two ones: one for training, the other one to realize tests
        # (to determine MAE or top-N items for example)
        train_set, test_set = train_test_split(data, test_size=.25)

        # Store MAEs values of the current experiment
        maes = []

        for k in k_list:
            algo = KNNBasic(k=k, sim_options=sim_options_knn)
            predictions = algo.fit(train_set).test(test_set)
            maes.append(mae(predictions))

        # Update the corresponding in the matrix of MAEs values
        m_maes[i, :] = maes

    # Best K information
    best_k_index = -1
    best_k_mae_value = 0

    average_maes_per_k = np.zeros(len(k_list))

    # Analyse MAEs values for each K and search the best one
    for k_index in range(0, len(k_list), 1):
        k_maes = m_maes[:, k_index]
        average_maes = sum(k_maes) / len(k_maes)
        average_maes_per_k[k_index] = average_maes
        if k_index == 0 or average_maes < best_k_mae_value:
            best_k_index = k_index
            best_k_mae_value = average_maes
        print("K={}, average MAEs={}".format(k_list[k_index], average_maes))

    # Display the best value of K
    if best_k_index >= 0:
        best_k = 40 + best_k_index * 4
        print("Best K: {}, average MAEs: {}".format(best_k, best_k_mae_value))
    else:
        print("No best K found")

    # Plot a curve of average MAEs per K
    plot_maes(k_list, average_maes_per_k)


def e1_a_search_k(data):
    # Use the "Pearson" similarity function for K-NN algorithm
    sim_options_knn = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }

    probabilistic_analysis_search_k(data, sim_options_knn)


def e3_knn(data):
    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=.25)

    # Use the "Pearson" similarity function for K-NN algorithm
    sim_options_knn = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }
    k = 50  # number of nearest-neighbours for KNN algorithm
    uid = str(145)  # the raw ID of user to recommend items

    # Use KNN for predictions
    algo = KNNBasic(k=k, sim_options=sim_options_knn)

    # Train the algo and then create predictions from test dataset defined previously
    predictions = algo.fit(train_set).test(test_set)

    # Retain only predictions of the targeted user
    user_predictions = get_user_predictions(predictions, uid)

    # Determine the top 10 items of user defined previously
    top_n_predictions = get_top_n_predictions(user_predictions, 10)
    print_recommendations(top_n_predictions)

    precision = get_precision(top_n_predictions)
    recall = get_recall(top_n_predictions, user_predictions)
    f1 = get_f1(precision, recall)
    print_recommendations_evaluation(precision, recall, f1)


def main():
    # Load the movielens-100k dataset
    data = Dataset.load_builtin('ml-100k')

    e1_a_search_k(data)


main()
