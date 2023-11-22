import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            # Нужно создать self.num_bags бутстраппированных выборок
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)

        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        preds = np.zeros(len(data))  # Создаем контейнер под предсказания
        for model in self.models_list:  # Каждая модель предсказывает данные
            preds += model.predict(data)
        # Усредняем по моделям (т.е. по бутстрапированным выборкам)
        return preds / self.num_bags

    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        # Прогон по всем объектам датасета
        for idx in range(len(self.data)):
            sample = self.data[idx].reshape(1, -1)  # см. документацию: "-1" это любое число. Иначе говоря - нам нужен вектор
            models_predictions = []

            # Прогон по всем моделям
            for bag in range(self.num_bags):
                # выполнение условия /*which have not seen this object during training phase*/
                if idx not in self.indices_list[bag]:
                    models_predictions.append(float(self.models_list[bag].predict(sample)))

            # Напомню - /*where list i contains predictions for self.data[i] object from all models*/
            list_of_predictions_lists[idx] = models_predictions

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Your Code Here

        # Инициализация контейнера под out of back предсказания
        self.oob_predictions = np.zeros(len(self.data))

        # Прогон по всем объектам датасета
        for idx in range(len(self.data)):
            models_predictions = self.list_of_predictions_lists[idx]

            # Реализация условия /*If object has been used in all bags on training phase, return None instead of prediction*/
            if len(models_predictions) == 0:
                self.oob_predictions[idx] = None
            else:
                self.oob_predictions[idx] = sum(models_predictions) / len(models_predictions)
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        
        # mean без учета nan'ов
        return np.nanmean((self.target - self.oob_predictions) ** 2) # Your Code Here