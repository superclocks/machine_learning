import docclass
import feedfilter
import os
from sklearn.metrics import classification_report
def readFile(file_name):
    reader = open(file_name, 'r')
    lines = reader.readlines()
    content = ' '.join(lines)
    return content

def train(file_path, model_name):
    cl=docclass.naivebayes(docclass.getwords)
    cl.setdb(model_name)
    
    name_list = os.listdir(file_path)
    for label in name_list:
        if label in ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc']:
            print 'Training ' + model_name + 'using ' + label + '...' 
            label_path = file_path + '/' + label
            file_names = os.listdir(label_path)
            i = 0
            for file_name in file_names:
                train_file_path = label_path + '/' + file_name
                content = readFile(train_file_path)
                cl.train(content, label)
                if i > 100:
                    break
                i = i + 1
                
def classifier(file_path, model_name):
    cl=docclass.naivebayes(docclass.getwords)
    cl.setdb(model_name)
    
    #
    true_list = [] #save true result
    pred_list = [] #save prediction result
    name_list = os.listdir(file_path)
    for label in name_list:
        if label in ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc']:
            print 'Using ' + model_name + 'to predict...'
            label_path = file_path + '/' + label
            file_names = os.listdir(label_path)
            i = 0
            for file_name in file_names:
                test_file_path = label_path + '/' + file_name
                content = readFile(test_file_path)
                try:
                    r = cl.classify(content)
                    pred_list.append(r)
                    true_list.append(label)
                    print 'predicttion: ' + r + '->' + label
                except:
                    print 'Bad prediction...'
                if i > 50:
                    break
                i = i + 1
    return [pred_list, true_list]


if __name__ == '__main__':
    #train('./20news-bydate/20news-bydate-train', '20news.model')
    [pred_list, true_list] = classifier('./20news-bydate/20news-bydate-test', '20news.model')
    classification_report(true_list, pred_list)
    
    