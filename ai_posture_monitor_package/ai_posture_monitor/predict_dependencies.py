def get_groundtruth_from_image_name(image_name=''):
    #dic = {'stand':'stand', 'squat':'squat', 'sit':'sit', 'lying':'lie', 'lie':'lie', 'bend':'stand', 'fall':'lie', 'climb':'stand', 'get':'sit'}
    dic = {'stand':'stand', 'squat':'stand', 'sit':'sit', 'lying':'lie', 'lie':'lie', 'bend':'stand', 'fall':'lie', 'climb':'stand', 'get':'sit'}
    if image_name != '':
        for x in dic:
            if image_name.startswith(x):
                return dic[x]
    return None

def get_attr_of_features():
    return ['image_name', 'is_upright', 'percent_upright', 'stand_left', 'stand_right', 'percent_stand_left', 'percent_stand_right', 'sit_left', 'sit_right', 'percent_sit_left', 'percent_sit_right', 'lie_left', 'lie_right']

def predict_pose(features=[], features_df=None):
    # Use features list to make prediction
    label = None

    def pred_stand_to_lie(feature_list):
        label = None
        squat = 'squat'
        squat = 'stand'
        if feature_list['stand_left'] == 'standing' and feature_list['stand_right'] == 'standing':
            label = 'stand'
        elif feature_list['stand_left'] == 'squat' and feature_list['stand_right'] == 'squat':
            label = squat
        elif feature_list['percent_upright'] < 20 and feature_list['sit_left'] == 'sitting' and feature_list['sit_right'] == 'sitting' and feature_list['lie_left'] == 'lying' and feature_list['lie_right'] == 'lying':
            label = 'lie'
        elif feature_list['sit_left'] == 'sitting' and feature_list['sit_right'] == 'sitting':
            label = 'sit'
        elif feature_list['stand_left'] == 'standing' and feature_list['stand_right'] == 'squat':
            label = 'stand'
        elif feature_list['stand_right'] == 'standing' and feature_list['stand_left'] == 'squat':
            label = 'stand'

        elif max([feature_list['percent_stand_left'], feature_list['percent_stand_right']]) > 90 and min([feature_list['percent_sit_right'], feature_list['percent_sit_left']]) < 20 :
            label = 'stand'

        elif max([feature_list['percent_stand_left'], feature_list['percent_stand_right']]) > 70 and min([feature_list['percent_sit_right'], feature_list['percent_sit_left']]) < 20  and max([feature_list['percent_sit_right'], feature_list['percent_sit_left']]) < 70 :
            label = 'stand'

        elif feature_list['stand_left'] == 'standing' and feature_list['stand_right'] == 'uncertain_standing':
            label = 'stand'
        elif feature_list['stand_left'] == 'squat' and feature_list['stand_right'] == 'uncertain_squat':
            label = squat
        elif feature_list['sit_left'] == 'sitting' and feature_list['sit_right'] == 'uncertain_sitting':
            label = 'sit'
        elif feature_list['stand_left'] == 'standing' and feature_list['stand_right'] == 'uncertain_squat':
            label = 'stand'
        elif feature_list['stand_right'] == 'uncertain_standing' and feature_list['stand_left'] == 'squat':
            label = 'stand'

        elif feature_list['stand_right'] == 'standing' and feature_list['stand_left'] == 'uncertain_standing':
            label = 'stand'
        elif feature_list['stand_right'] == 'squat' and feature_list['stand_left'] == 'uncertain_squat':
            label = squat
        elif feature_list['sit_right'] == 'sitting' and feature_list['sit_left'] == 'uncertain_sitting':
            label = 'sit'
        elif feature_list['stand_right'] == 'standing' and feature_list['stand_left'] == 'uncertain_squat':
            label = 'stand'
        elif feature_list['stand_left'] == 'uncertain_standing' and feature_list['stand_right'] == 'squat':
            label = 'stand'

        elif feature_list['stand_left'] == 'standing' or feature_list['stand_right'] == 'standing':
            label = 'stand'
        elif feature_list['sit_left'] == 'sitting' or feature_list['sit_right'] == 'sitting':
            label = 'sit'
        elif feature_list['stand_left'] == 'squat' or feature_list['stand_right'] == 'squat':
            label = squat
        elif feature_list['lie_left'] == 'lying' and feature_list['lie_right'] == 'lying':
            label = 'lie'
        elif feature_list['lie_left'] == 'lying' or feature_list['lie_right'] == 'lying':
            label = 'lie'
        elif feature_list['stand_left'] == 'uncertain_standing' and feature_list['stand_right'] == 'uncertain_standing':
            label = 'stand'
        elif feature_list['stand_left'] == 'uncertain_squat' and feature_list['stand_right'] == 'uncertain_squat':
            label = squat
        elif feature_list['stand_left'] == 'uncertain_standing' and feature_list['stand_right'] == 'uncertain_squat':
            label = 'stand'
        elif feature_list['stand_right'] == 'uncertain_standing' and feature_list['stand_left'] == 'uncertain_squat':
            label = 'stand'
        elif feature_list['sit_left'] == 'uncertain_sitting' and feature_list['sit_right'] == 'uncertain_sitting':
            label = 'sit'
        elif feature_list['stand_left'] == 'uncertain_standing' or feature_list['stand_right'] == 'uncertain_standing':
            label = 'stand'
        elif feature_list['stand_left'] == 'uncertain_squat' or feature_list['stand_right'] == 'uncertain_squat':
            label = squat
        elif feature_list['sit_left'] == 'uncertain_sitting' or feature_list['sit_right'] == 'uncertain_sitting':
            label = 'sit'
        elif feature_list['lie_left'] == 'uncertain_lying' and feature_list['lie_right'] == 'uncertain_lying':
            label = 'lie'
        elif feature_list['lie_left'] == 'uncertain_lying' or feature_list['lie_right'] == 'uncertain_lying':
            label = 'lie'
        return label

    if len(features) > 0:
        feature_list = dict(zip(get_attr_of_features(), features))
    elif features_df is not None:
        feature_list = features_df

    if feature_list is not None:
        if feature_list['is_upright']:
            # stand or sit
            label = pred_stand_to_lie(feature_list)
        elif feature_list['stand_left'] == 'standing' and feature_list['stand_right'] == 'standing':
            label = 'stand'
        else:
            # likely lie
            if feature_list['lie_left'] == 'lying' and feature_list['lie_right'] == 'lying':
                label = 'lie'
            elif feature_list['lie_left'] == 'lying' or feature_list['lie_right'] == 'lying':
                label = 'lie'
            else:
                label = pred_stand_to_lie(feature_list)

    return label