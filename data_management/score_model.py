import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import graphlab as gl

class RecScorer(object):
    '''
    Scores on precision, recall, and F1 Score.
    '''

    def __init__(self, recommender, sf, test_set, new_item_data=None, name=None, color=None, axes=None, i=1):
        '''
        Input: trained graphlab recommender object
                Graphlab SFrame of Test Data
        '''
        self.recommender = recommender
        self.n_users = recommender.num_users
        self.all_data = sf
        self.sf = test_set
        self.new_item_data = new_item_data
        self.i = i
        if axes.any():
            self.axes = axes
            self.name = name
            self.color = color
        else:
            self.fig, self.axes = plt.subplots(1,2, figsize=(14,6))
            self.name=None
            self.color='r'

    def score_precision_recall(self, plot=True):
        '''
        from all data:
            -> mean rating for each user
            -> cut off recipes that are not rated highly
                - cut off is mean + 1 std of each user
            ->give 1 or 0 in 'bool' column

        test_data:
            -> predict on test_data recipes
            -> predictions that are high rated must:
                - have predicted score above cut-off dependent on each user
                -> these are assigned as 1 in bool column

        input to graphlab scorers and plot:

        "The precision is the proportion of recommendations that are good recommendations, and recall is the proportion of good recommendations that appear in top recommendations."
        '''
        df = self._get_bools()
        self.precision = gl.evaluation.precision(targets=gl.SArray(df['bool']), predictions=gl.SArray(df['rating_bool']))
        self.recall = gl.evaluation.recall(targets=gl.SArray(df['bool']), predictions=gl.SArray(df['rating_bool']))

        if plot:
            self.plot_rmse()
            self.plot_precision_recall()

    def _get_bools(self):
        df = self.all_data.to_dataframe()

        sorted_df = df.sort_values(by=['user_id', 'rating'], ascending=False)
        mean_df = df.groupby('user_id').mean()
        std_df = df.groupby('user_id').std().fillna(0)

        sorted_df = sorted_df.join(mean_df, on='user_id', rsuffix='_mean')
        sorted_df = sorted_df.join(std_df, on='user_id', rsuffix='_std')
        sorted_df['threshold'] = sorted_df['rating_mean'] + sorted_df['rating_std']
        sorted_df['bool'] = sorted_df['rating'] >= sorted_df['threshold']
        # sorted_df['bool'] = sorted_df['rating'] >= sorted_df['rating_mean']
        sorted_df['bool'] = sorted_df['bool'].map({True: 1, False: 0})

        sf = self.sf.copy()
        predictions = self.recommender.predict(sf, new_item_data=self.new_item_data)
        predictions = sf.add_column(predictions, 'rating_pred').to_dataframe()
        predictions.drop(['rating'], axis=1, inplace=True)

        sorted_df.astype('float64')
        predictions.astype('float64')
        sorted_df = sorted_df.merge(predictions, on=['user_id', 'recipe_id'])
        sorted_df['rating_bool'] = sorted_df['rating_pred'] >= sorted_df['rating_mean']
        sorted_df['rating_bool'] = sorted_df['rating_bool'].map({True: 1, False: 0})
        return sorted_df

    def _my_rmse(self):
        '''
        predict on test_set -> match with actual rating -> group by user_id for scoring
        -> find rmse for average of each user's ratings -> return overall rmse
        '''
        # predictions = self.recommender.predict(self.sf)
        # sf = self.sf.add_column(predictions, 'rating_pred')
        df = self.sf.to_dataframe().groupby('user_id').mean()
        num_users = df.shape[0]
        df['rmse'] = np.sqrt(((df['rating'] - df['rating_pred'])**2)/ num_users)
        return df['rmse'].mean()

    def plot_rmse(self):
        # self.axes[1].axhline(self._my_rmse(), color='b', label='My RMSE')
        # self.axes[1].axhline(self.recommender['training_rmse'], color='r', linestyle='--', label='Train RMSE')
        test_rmse = gl.evaluation.rmse(targets=self.sf['rating'], predictions=self.recommender.predict(self.sf, new_item_data=self.new_item_data))
        print test_rmse
        self.axes[1].bar(self.i, test_rmse, align='center', color=self.color, alpha=0.8)
        # self.axes[1].scatter(1, test_rmse, color=self.color, s = 50, label=self.name)
        self.axes[1].annotate('{0:.2f}'.format(test_rmse), xy=(self.i-0.05, test_rmse+0.005), textcoords='data')
        self.axes[1].set_xticks([self.i], self.name)
        self.axes[1].set_ylabel('RMSE')
        # self.axes[1].legend(loc=2)
        # self.axes[1].set_ylim(0, 0.02)
        self.axes[1].set_title('Recommender Test RMSE')

    def plot_precision_recall(self):
        '''
        plots precision on x-axis
        plots recall on y-axis

        '''
        # nums, precision, recall = zip(*npr)
        # c = plt.get_cmap('viridis')
        # colors = [c(0.15*float(i)/len(self.slices)) for i in self.slices]
        self.axes[0].scatter(self.precision, self.recall, color=self.color, s=50, alpha=1.0, label=self.name)

        self.axes[0].set_xlabel('Precision')
        self.axes[0].set_ylabel('Recall')
        self.axes[0].set_title('Precision - Recall Plot')
        self.axes[0].legend(loc='best')
        self.axes[0].set_xlim(0.0, 1.0), self.axes[0].set_ylim(0.0, 1.05)

        # self.axes[0] = self._make_f1_lines()

    def _make_f1_lines(self):
        precision_values = np.linspace(0.0, 1.0, num=500)[1:]
        f1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for f in f1_values:
            points = []
            for p in precision_values:
                recall = self._calc_recall(f, p)
                if recall > 0 and recall <=1.5:
                    points.append((p, recall))
            recall_pts, precision_pts = zip(*points)
            self.axes[0].plot(recall_pts, precision_pts , "--", color="gray", linewidth=0.5)
            self.axes[0].annotate(r"$f1=%.1f$" % f, xy=(recall_pts[-10], precision_pts[-10]), xytext=(recall_pts[-10]-0.08, precision_pts[-10]-0.03), size="small", color="gray")

    def _calc_recall(self, f, p):
        return f * p / (2 * p - f)

if __name__ == '__main__':
    'load data train test model'
    df = pd.read_pickle('pkls/data.pkl')
    sf = gl.SFrame(df)

    'decide train test split type'
    train_set, test_set = sf.random_split(fraction=0.75, seed=42)
    # train_set, test_set = gl.recommender.util.random_split_by_user(sf, user_id='user_id', item_id='recipe_id', item_test_proportion=.25, random_seed=42)

    'create or load models'
    # model = gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=None)
    # model.save('scorer_test_model')

    regularization_vals = [0.001, 0.0001, 0.00001, 0.000001]
    models = [gl.factorization_recommender.create(train_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=None, max_iterations=50, num_factors=8, regularization=r)for r in regularization_vals]
    # model = gl.load_model('scorer_test_model')
    test = ScoreMany(recommenders=models, sf=sf, test_set=test_set)
    test.plot_score_all()
    # test.score_precision_recall()
    # test._make_f1_lines
