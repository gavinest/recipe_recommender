import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import graphlab as gl

class RecScorer(object):
    '''
    Scores on precision, recall, and F1 Score.
    '''

    def __init__(self, recommender, test_set):
        '''
        Input: trained graphlab recommender object
                Graphlab SFrame of Test Data
        '''
        self.recommender = recommender
        self.n_users = recommender.num_users
        self.sf = test_set

    def get_precision_recall(self, slices=range(10, 110, 10), plot_precision_recall=True):
        self.slices = slices
        npr = []
        for i in slices:
            sf_slice = self.sf[:i]
            predictions = self.recommender.predict(sf_slice)
            precision, recall = self._get_score_metrics(predictions, sf_slice)
            npr.append([i, precision, recall])
        if plot_precision_recall:
            self.plot_precision_recall(npr)
        self.precision_recall_scores = npr
        return npr

    def _get_score_metrics(self, predictions, sf_slice):
        predictions = np.round(predictions)
        predictions = gl.SArray(data=predictions, dtype=int)

        precision = gl.evaluation.precision(targets=sf_slice['rating'], predictions=predictions)
        recall = gl.evaluation.recall(targets=sf_slice['rating'], predictions=predictions)
        # f1 = gl.evaluation.f1_score(targets=sf_slice['rating'], predictions=predictions)
        return precision, recall

    def plot_precision_recall(self, npr):
        '''
        plots precision on x-axis
        plots recall on y-axis

        '''
        nums, precision, recall = zip(*npr)
        c = plt.get_cmap('viridis')
        colors = [c(0.15*float(i)/len(self.slices)) for i in self.slices]

        fig, ax = plt.subplots(1, figsize=(8,8))
        for i, n in enumerate(nums):
            ax.scatter(precision[i], recall[i], color=colors[i], s=50, alpha=0.8, label='Size - {}'.format(n))

        ax = self._make_f1_lines(ax=ax)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        ax.set_title('Precision - Recall Plot')
        ax.legend(loc='best')
        ax.set_xlim(0.0, 1.0), ax.set_ylim(0.0, 1.05)
        self.ax = ax
        # plt.savefig('precision_recall_data10k_factrec.jpg')
        # plt.show()

    def _make_f1_lines(self, ax):

        precision_values = np.linspace(0.0, 1.0, num=500)[1:]
        f1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for f in f1_values:
            points = []
            for p in precision_values:
                recall = self._calc_recall(f, p)
                if recall > 0 and recall <=1.5:
                    points.append((p, recall))
            recall_pts, precision_pts = zip(*points)
            ax.plot(recall_pts, precision_pts , "--", color="gray", linewidth=0.5)
            ax.annotate(r"$f1=%.1f$" % f, xy=(recall_pts[-10], precision_pts[-10]), xytext=(recall_pts[-10]-0.06, precision_pts[-10]-0.02), size="small", color="gray")
        return ax

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
    model = gl.factorization_recommender.create(test_set, user_id='user_id', item_id='recipe_id', target='rating', item_data=None)
    # model.save('scorer_test_model')
    # model = gl.load_model('scorer_test_model')
    test = RecScorer(recommender=model, test_set=test_set)
    test.rmse_of_top_percent()
