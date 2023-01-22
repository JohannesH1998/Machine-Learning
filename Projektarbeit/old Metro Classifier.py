class MetroClassifier:

    __labels = None

    def __init__(self, df, var = 0.95, window_size = 10, excluded_columns = ['timestamp', 'gpsLong', 'gpsLat', 'gpsSpeed', 'gpsQuality']):
        self.var = var
        self.window_size = window_size
        self.excluded_columns = excluded_columns
        self.df = df

        #check if var is between 0 and 1
        if self.var < 0 or self.var > 1:
            raise ValueError('var must be between 0 and 1')
            
        #all colums without Label
        self.columns = [x for x in df.columns if x not in excluded_columns]

        cols_without_label = self.columns.copy()
        cols_without_label.remove('Label')

        self.decision_tree = None


    def __getRollingWindowDf(self, df, window_size, excluded_columns):
        self.__labels = df['Label']
        columns  = [x for x in df.columns if x not in excluded_columns]
        columns.append('Label')

        operations = ['mean', 'std', 'min', 'max']
        #generate a dict, where the keys are the column names and the values are the operations that should be performed on the column
        operations_dict = {}
        for column in columns:
            operations_dict[column] = operations
            #keep the labels
            operations_dict['Label'] = ['max']

        df_rolling = df.rolling(window_size).agg(operations_dict)

        #drop all columns that have NaN values
        df_rolling = df_rolling.dropna()

        #flatten df_rolling
        df_rolling.columns = ['_'.join(col) for col in df_rolling.columns]
        #rename Label_min to Label
        df_rolling = df_rolling.rename(columns={'Label_max': 'Label'})

        print("ROLLING", df_rolling.head())
        return df_rolling
    
    def __scaleDf(self, df):
        scaler = MinMaxScaler()

        #cut label column from df

        df = df.drop(['Label'], axis=1)

        data_rescaled = scaler.fit_transform(df)

        print("SCALED", data_rescaled)

        return data_rescaled
    
    def __performPca(self, df, index):      

        pca = PCA(n_components=self.var, random_state=0)
   
        print("PCA-fit", df)

        #pca_columns = ['PCA%i' % i for i in range(pca.n_components_)]

        #print("PCA-columns", pca_columns)

        print("PCA-index", index)

        transformed = pca.fit_transform(df)
        print("PCA-transformed", transformed)

        tmp = pd.DataFrame(transformed, ['PCA%i' % i for i in range(pca.n_components_)], index=index)

        #add label column again
        tmp['Label'] = self.__labels

        print("PCA-labels", self.__labels)
        print("PCA-res", tmp)

        return tmp
    
    def fit(self):
        df = self.__getRollingWindowDf(self.df, self.window_size, self.excluded_columns)
        index = df.index
        df = self.__scaleDf(df)
        df = self.__performPca(df, index)

        self.decision_tree = DecisionTreeBinaryClassifier(columns = [x for x in df.columns if x not in ['Label']], random_state = 0, max_depth=3)
        self.decision_tree.fit(df=df, label='Label')

    def score(self):
        return self.decision_tree.score()
    
    def confusionMatrix(self):
        return self.decision_tree.confusionMatrix()
    
    def corellationMatrix(self):
        return self.decision_tree.corellationMatrix()
        