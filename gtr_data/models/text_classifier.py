# CLasses

#One class for text classification based on text inputs

class TextClassification():
    '''
    This class takes a corpus (could be a list of strings or a tokenised corpus) and a target (could be multiclass or single class).
    
    When it is initialised it vectorises the list of tokens using sklearn's count vectoriser.
    
    It has a grid search method that takes a list of models and parameters and trains the model.
    
    It returns the output of grid search for diagnosis
    
    '''
    
    def __init__(self,corpus,target):
        '''
        
        Initialise. The class will recognise if we are feeding it a list of strings or a list of
        tokenised documents and vectorise accordingly. 
        
        It will also recognise is this a multiclass or one class problem based on the dimensions of the target array
        
        Later on, it will use control flow to modify model parameters depending on the type of data we have
        
        '''
        
        #Is this a multiclass classification problem or a single class classification problem?
        if target.shape[1]>1:
            self.mode = 'multiclass'
            
        else:
            self.mode = 'single_class'
    
    
        #Store the target
        self.Y = target
    
        #Did we feed the model a bunch of strings or a list of tokenised docs? If the latter, we clean and tokenise.
        
        if type(corpus[0])==str:
            corpus = CleanTokenize(corpus).clean().bigram().tokenised
            
        #Turn every list of tokens into a string for count vectorising
        corpus_string =  [' '.join(words) for words in corpus]
        
        
        #And then we count vectorise in a hacky way.
        count_vect = CountVectorizer(stop_words='english',min_df=5).fit(corpus_string)
        
        #Store the features
        self.X = count_vect.transform(corpus_string)
        
        #Store the count vectoriser (we will use it later on for prediction on new data)
        self.count_vect = count_vect
        
    def grid_search(self,models):
        '''
        The grid search method takes a list with models and their parameters and it does grid search crossvalidation.
        
        '''
        
        #Load inputs and targets into the model
        Y = self.Y
        X = self.X
        
        if self.mode=='multiclass':
            '''
            If the model is multiclass then we need to add some prefixes to the model paramas
            
            '''
        
            for mod in models:
                #Make ovr
                mod[0] = OneVsRestClassifier(mod[0])
                
                #Add the estimator prefix
                mod[1] = {'estimator__'+k:v for k,v in mod[1].items()}
                
        
        #Container with results
        results = []

        #For each model, run the analysis.
        for num,mod in enumerate(models):
            print(num)

            #Run the classifier
            clf = GridSearchCV(mod[0],mod[1])

            #Fit
            clf.fit(X,Y)

            #Append results
            results.append(clf)
        
        self.results = results
        return(self)

    
#Class to visualise the outputs of multilabel models.

#I call it OrangeBrick after YellowBrick, the package for ML output visualisation 
#(which currently doesn't support multilabel classification)


class OrangeBrick():
    '''
    This class takes a df with the true classes for a multilabel classification exercise and produces some charts visualising findings.
    
    The methods include:
    
        .confusion_stack: creates a stacked barchart with the confusion matrices stacked by category, sorting classes by performance
        .prec_rec: creates a barchart showing each class precision and recall;
        #Tobe done: Consider mixes between classes?
    
    '''
    
    def __init__(self,true_labels,predicted_labels,var_names):
        '''
        Initialise with a true labels, predicted labels and the variable names
        '''
         
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.var_names = var_names
    
    def make_metrics(self):
        '''
        Estimates performance metrics (for now just confusion charts by class and precision/recall scores for the 0.5 
        decision rule.
        
        '''
        #NB in a confusion matrix in SKlearn the X axis indicates the predicted class and the Y axis indicates the ground truth.
        #This means that:
            #cf[0,0]-> TN
            #cf[1,1]-> TP
            #cf[0,1]-> FN (prediction is false, groundtruth is true)
            #cf[1,0]-> FP (prediction is true, ground truth is false)



        #Predictions and true labels
        true_labels = self.true_labels
        pred_labels = self.predicted_labels

        #Variable names
        var_names = self.var_names

        #Store confusion matrices
        score_store = []


        for num in np.arange(len(var_names)):

            #This is the confusion matrix
            cf = confusion_matrix(pred_labels[:,num],true_labels[:,num])

            #This is a melted confusion matrix
            melt_cf = pd.melt(pd.DataFrame(cf).reset_index(drop=False),id_vars='index')['value']
            melt_cf.index = ['true_negative','false_positive','false_negative','true_positive']
            melt_cf.name = var_names[num]
            
            #Order variables to separate failed vs correct predictions
            melt_cf = melt_cf.loc[['true_positive','true_negative','false_positive','false_negative']]

            #We are also interested in precision and recall
            prec = cf[1,1]/(cf[1,1]+cf[1,0])
            rec = cf[1,1]/(cf[1,1]+cf[0,1])

            prec_rec = pd.Series([prec,rec],index=['precision','recall'])
            prec_rec.name = var_names[num]
            score_store.append([melt_cf,prec_rec])
    
        self.score_store = score_store
        
        return(self)
    
    def confusion_chart(self,ax):
        '''
        Plot the confusion charts
        
        
        '''
        
        #Visualise confusion matrix outputs
        cf_df = pd.concat([x[0] for x in self.score_store],1)

        #This ranks categories by the error rates
        failure_rate = cf_df.apply(lambda x: x/x.sum(),axis=0).loc[['false' in x for x in cf_df.index]].sum().sort_values(
            ascending=False).index

        
        #Plot and add labels
        cf_df.T.loc[failure_rate,:].plot.bar(stacked=True,ax=ax,width=0.8,cmap='Accent')

        ax.legend(bbox_to_anchor=(1.01,1))
        #ax.set_title('Stacked confusion matrix for disease areas',size=16)
    
    
    def prec_rec_chart(self,ax):
        '''
        
        Plot a precision-recall chart
        
        '''
    

        #Again, we sort them here to assess model performance in different disease areas
        prec_rec = pd.concat([x[1] for x in self.score_store],1).T.sort_values('precision')
        prec_rec.plot.bar(ax=ax)

        #Add legend and title
        ax.legend(bbox_to_anchor=(1.01,1))
        #ax.set_title('Precision and Recall by disease area',size=16)