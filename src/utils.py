import os
import re
import json
import pandas as pd


def create_dataframe(account_data_list):
    data_list = []
    for account_data in account_data_list:
        user_follower_count = account_data["userFollowerCount"]
        user_following_count = account_data["userFollowingCount"]
        follower_following_ratio = user_follower_count / max(1, user_following_count)

        temp_dataframe = pd.Series({
            "user_media_count": account_data["userMediaCount"],
            "user_follower_count": account_data["userFollowerCount"],
            "user_following_count": account_data["userFollowingCount"],
            "user_has_profil_pic": account_data["userHasProfilPic"],
            "user_is_private": account_data["userIsPrivate"],
            "follower_following_ratio": follower_following_ratio,
            "user_biography_length": account_data["userBiographyLength"],
            "username_length": account_data["usernameLength"],
            "username_digit_count": account_data["usernameDigitCount"],
            "is_fake": account_data["isFake"]
        })
        data_list.append(temp_dataframe)

    dataframe = pd.DataFrame(data_list)
    return dataframe


def import_data(dataset_path):
    with open(dataset_path + "/fakeAccountData.json") as json_file:
        fake_account_data = json.load(json_file)
    with open(dataset_path + "/realAccountData.json") as json_file:
        real_account_data = json.load(json_file)

    fake_account_dataframe = create_dataframe(fake_account_data)
    real_account_dataframe = create_dataframe(real_account_data)
    merged_dataframe = pd.concat([fake_account_dataframe, real_account_dataframe], ignore_index=True)
    data = dict({"dataframe": merged_dataframe})

    return data


def get_reason_for_classification(tree, feature_names, instance):
    node_indicator = tree.decision_path([instance])
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    reasons = []

    # Iterate over the nodes in the decision path
    for node_index in node_indicator.indices:
        # Skip leaf nodes
        if feature[node_index] != -2:
            feature_name = feature_names[feature[node_index]]
            threshold_value = threshold[node_index]
            feature_value = instance[feature[node_index]]

            if feature_value <= threshold_value:
                reasons.append(f"{feature_name} <= {threshold_value}")
            else:
                reasons.append(f"{feature_name} > {threshold_value}")

    return reasons