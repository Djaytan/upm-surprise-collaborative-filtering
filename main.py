import numpy as np
from matplotlib import pyplot
from surprise import Dataset
from surprise import KNNBasic
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


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


def dichotomy_search_k(data, sim_options_knn, test_size):
    """
    Permits to search the best K value which minimize the MAE value with the dichotomy approach.
    The name of the method refer to the usage of the method more than its reel functioning.

    :param data: The dataset used in experiments.
    :param sim_options_knn: The options of the similarity function of the K-NN algorithm.
    """
    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=test_size)

    # Adjust range values according to the experience to realize
    k_list = list(range(50, 100, 10))

    # Store MAEs values to display them later
    maes = []

    for k in k_list:
        algo = KNNBasic(k=k, sim_options=sim_options_knn)
        predictions = algo.fit(train_set).test(test_set)
        maes.append(mae(predictions))

    plot_maes(k_list, maes)


def probabilistic_analysis_search_k(data, sim_options_knn, test_size):
    """
    Searches the best K value which minimize the MAE with probabilistic analysis approach.

    :param data: The dataset used in experiments.
    :param sim_options_knn: The options of the similarity function of the K-NN algorithm.
    """
    # Number of experiments (>= 30 to permit quantitative analysis)
    n = 1

    # We focus on the range of values of K where MAEs remain unstable
    k_list = list(range(40, 200, 1))

    # Matrix of MAEs values: row correspond to an experiment and column to a K value
    # A way to read a cell: for the experiment n°X with K=Y the MAE value is equal to Z (the value of the cell)
    m_maes = np.zeros((n, len(k_list)))

    for i in range(0, n, 1):
        # Split dataset in two ones: one for training, the other one to realize tests
        # (to determine MAE or top-N items for example)
        train_set, test_set = train_test_split(data, test_size=test_size)

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


def e1_search_k(data):
    """
    The exercise 1.a of the assignment.

    :param data: The dataset used in experiments.
    """
    # Use the "Pearson" similarity function for K-NN algorithm
    sim_options_knn = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }

    probabilistic_analysis_search_k(data, sim_options_knn, .25)


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
    return get_nb_true_positives(top_n_predictions) / max(get_nb_relevant_elements(user_predictions), 1)


def get_f1(precision, recall):
    """
    Gets F1 measure for evaluating predictions by combining precision and recall ones together.

    :param precision: The precision of the predictions.
    :param recall: The recall of the predictions.
    :return: The F1 measure according to the specified precision and recall values.
    """
    if precision == 0 and recall == 0:
        return 0
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


def get_raw_users_ids(predictions):
    """
    Gets from the specified predictions all existing users.

    :param predictions: The predictions of ratings to items of users.
    :return: The existing users from the specified predictions.
    """
    users = set()
    for prediction in predictions:
        users.add(prediction.uid)
    return users


def evaluate_recommendations_users(predictions):
    # Recover information about all the users of the system
    raw_users_ids = get_raw_users_ids(predictions)
    nb_users = len(raw_users_ids)

    # Range of N values to use for experiments
    max_n = 100
    n_list = list(range(10, max_n + 1, 1))

    # For each N value we store the average value of precision, recall and F1 in order to plot these values
    precisions_over_n = np.empty(0)
    recalls_over_n = np.empty(0)
    f1s_over_n = np.empty(0)

    # Experiments evaluation of recommendations over different values of N
    for n_top_items in n_list:
        print("N={}/{}".format(n_top_items, max_n))

        # Store precisions, recalls and F1s values for all recommendations of the users
        precisions = np.empty(0)
        recalls = np.empty(0)
        f1s = np.empty(0)

        # Iterate over each user to evaluate each recommendations
        for raw_user_id in raw_users_ids:
            # Retain only predictions of the targeted user
            user_predictions = get_user_predictions(predictions, raw_user_id)

            # Determine the top N items of user defined previously
            top_n_predictions = get_top_n_predictions(user_predictions, n_top_items)

            # Calculate precision, recall and F1 for the obtained recommendations
            precision = get_precision(top_n_predictions)
            recall = get_recall(top_n_predictions, user_predictions)
            f1 = get_f1(precision, recall)

            precisions = np.append(precisions, precision)
            recalls = np.append(recalls, recall)
            f1s = np.append(f1s, f1)

        # Calculate average precision, recall and F1 for the current value of N
        average_precision = sum(precisions) / nb_users
        average_recall = sum(recalls) / nb_users
        average_f1 = sum(f1s) / nb_users

        # Store these average measures for this current value of N
        precisions_over_n = np.append(precisions_over_n, average_precision)
        recalls_over_n = np.append(recalls_over_n, average_recall)
        f1s_over_n = np.append(f1s_over_n, average_f1)

    # Plot precisions, recalls and F1s values over the evolution of N
    pyplot.plot(n_list, precisions_over_n)
    pyplot.title('Precision evolution according to the increase of recommended items')
    pyplot.xlabel('Top N items recommended')
    pyplot.ylabel('Precision of recommendations')
    pyplot.show()

    pyplot.plot(n_list, recalls_over_n)
    pyplot.title('Recall evolution according to the increase of recommended items')
    pyplot.xlabel('Top N items recommended')
    pyplot.ylabel('Recalls of recommendations')
    pyplot.show()

    pyplot.plot(n_list, f1s_over_n)
    pyplot.title('F1 evolution according to the increase of recommended items')
    pyplot.xlabel('Top N items recommended')
    pyplot.ylabel('F1s of recommendations')
    pyplot.show()


def e3_knn(data):
    """
    The exercise 3 of the assignment (specific to the K-NN algorithm).

    :param data: The dataset used in experiments.
    """
    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=.25)

    # Use the "Pearson" similarity function for K-NN algorithm
    sim_options_knn = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }
    k = 60  # Number of nearest-neighbours for KNN algorithm

    # Use KNN for predictions
    algo = KNNBasic(k=k, sim_options=sim_options_knn)

    # Train the algo and then create predictions from test dataset defined previously
    predictions = algo.fit(train_set).test(test_set)

    # Evaluate recommendations for all users with with precision, recall and F1
    evaluate_recommendations_users(predictions)


def main():
    # Load the movielens-100k dataset
    data = Dataset.load_builtin('ml-100k')

    e1_search_k(data)


main()
