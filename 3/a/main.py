from surprise import Dataset
from surprise import KNNBasic
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
    nb_true_positives = 0
    for prediction in top_n_predictions:
        if prediction.r_ui >= 4:
            nb_true_positives = nb_true_positives + 1
    return nb_true_positives

def get_nb_relevant_elements(user_predictions):
    """
    Gets the number of relevant elements among all not-rating yet items.
    We expect here that relevant elements are those rated with 4 or 5 stars.

    :param: User's predictions (which contains reels items' ratings).
    :return: The number of relevant elements for recommendations.
    """
    nb_relevant_elements = 0
    for user_prediction in user_predictions:
        if user_prediction.r_ui >= 4:
            nb_relevant_elements = nb_relevant_elements + 1
    return nb_relevant_elements


def get_precision(top_n_predictions):
    return get_nb_true_positives(top_n_predictions) / len(top_n_predictions)

def get_recall(top_n_predictions, user_predictions):
    return get_nb_true_positives(top_n_predictions) / get_nb_relevant_elements(user_predictions)

def get_f1(precision, recall):
    """
    Gets F1 measure for evaluating recommendations by combining precision and recall ones together.

    :param precision: The precision of recommendations.
    :param recall: The recall of recommendations.
    :return: The F1 measure according to the specified precision and recall values.
    """
    return (2 * precision * recall) / (precision + recall)

def display_recommendations(top_predictions):
    """
    Display recommendations for the user associated this specified top predictions. It is assume here that all
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

def display_recommendations_evaluation(precision, recall, f1):
    print("Evaluation of recommendations")
    print("Precision =", round(precision, 3))
    print("Recall =", round(recall, 3))
    print("F1 =", round(f1, 3))


def main():
    #############
    ### Setup ###
    #############

    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')
    sim_options_knn = {
        'name': "pearson",
        'user_based': True  # compute similarities between users
    }
    k = 50  # number of nearest-neighbours for KNN algorithm
    uid = str(145)  # the raw ID of user to recommend items

    ###################
    ### Calculation ###
    ###################

    # Split dataset in two ones: one for training, the other one to realize tests
    # (to determine MAE or top-N items for example)
    train_set, test_set = train_test_split(data, test_size=.25)

    # Use KNN for predictions
    algo = KNNBasic(k=k, sim_options=sim_options_knn)

    # Train the algo and then create predictions from test dataset defined previously
    predictions = algo.fit(train_set).test(test_set)

    # Retain only predictions of the targeted user
    user_predictions = get_user_predictions(predictions, uid)

    # Determine the top 10 items of user defined previously
    top_n_predictions = get_top_n_predictions(user_predictions, 10)
    display_recommendations(top_n_predictions)

    precision = get_precision(top_n_predictions)
    recall = get_recall(top_n_predictions, user_predictions)
    f1 = get_f1(precision, recall)
    display_recommendations_evaluation(precision, recall, f1)


main()