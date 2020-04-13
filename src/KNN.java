import java.io.*;
import java.util.*;

public class KNN {


    public static void main(String[] args) {

        String trainingFile = args[0];
        String testFile = args[1];


        // set K as 1 unless specified otherwise
        int k = 1;
        if (args.length == 3) k = Integer.parseInt(args[2]);

        // set up data structures for algorithm
        ArrayList<double[]> testData = parseFile(testFile);
        ArrayList<double[]> trainingData = parseFile(trainingFile);
        assert testData != null;
        assert trainingData != null;

        // set up ranges of features for normalising the Euclidean algorithm later
        double[] ranges = findRanges(trainingData);

        double  accuracy = 0;

        // run each instance through the algorithm - if it guesses the label correctly, then increase accuracy
        int i = 0;
        double  j = 0;
        for (double[] testRow: testData){
            int labelGuess =  knnAlgorithm(testRow, trainingData, k, ranges);
            System.out.printf("Instance %d:- \t Label: %d | Guess: %d\n", i, (int)testRow[13], labelGuess);
            i++;
            if (labelGuess == testRow[13]) j++;
        }
        double acc = (j)/(testData.size())*100;

        System.out.printf("\n-------------------\n" +
                "K value: %d\n" +
                "Correct Classes: %d\n" +
                "Total Classes: %d\n" +
                "Percentage Correct: %f%%", k, (int)j, testData.size(), acc);

    }

    private static double[] findRanges(ArrayList<double[]> trainingData) {
        double[] rangeArray = new double [trainingData.get(0).length - 1]; // Ignoring label at the end

        for (int i = 0; i < rangeArray.length; i++) {
            double max = 0, min = 0;
            for (double[] trainingDatum : trainingData) {
                double num = trainingDatum[i];
                if (num > max) {
                    max = num;
                } else if (num < min) {
                    min = num;
                }
            }
            rangeArray[i] = max - min + 1;
        }
        return rangeArray;
    }

    private static int knnAlgorithm(double[] testRow, ArrayList<double[]> trainingData, int k, double[] ranges) {

        // initialise our neighbour array
        Tuple[] neighbours = new Tuple[k];

        // find the Euclidean distance between the test and the rest of the training set
        for (double[] trainingRow: trainingData) {
            double dist = euclid(testRow, trainingRow, ranges);

            //search through neighbour set for larger distances to replace
            for (int i = 0; i < k; i++){
                if (neighbours[i] == null || dist < neighbours[i].getDist()) {
                    neighbours[i] = new Tuple(dist, trainingRow[13]);
                    break;
                }
            }
        }

        return findLabel(neighbours);
    }

    private static int findLabel(Tuple[] neighbours) {
        int[] count = {0, 0, 0};
        int guess = 0;

        for (Tuple neighbour : neighbours) {
            switch ((int) neighbour.getLabel()) {
                case 1:
                    count[0]++;
                    break;
                case 2:
                    count[1]++;
                    break;
                case 3:
                    count[2]++;
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + (int) neighbour.getLabel());
            }
        }

        if (count[0] > count[1] && count[0] > count[2]) guess = 1;
        else if (count[1] > count[0] && count[1] > count[2]) guess = 2;
        else if (count[2] > count[0] && count[2] > count[1]) guess = 3;
        return guess;
    }


    private static double euclid(double[] testRow, double[] trainingRow, double[] ranges) {
        assert testRow.length == trainingRow.length;
        double sum = 0.0;

        // for every feature except the last one (class label)
        for (int i = 0; i < testRow.length -1; i++) {

            // Euclidean distance equation normalised by range values of each feature
            sum += Math.pow(testRow[i] - trainingRow[i], 2) / Math.pow(ranges[i], 2);

        }
        return Math.sqrt(sum);
    }

    private static ArrayList<double[]> parseFile(String file) {

        FileInputStream stream = null;
        try {
            stream = new FileInputStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        assert stream != null;

        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String strLine;
        try {
            ArrayList<double[]> dataArray = new ArrayList<>();

            reader.readLine(); //  Ignore first line

            while ((strLine = reader.readLine()) != null) {
                String[] rowString = strLine.split(" ");
                double[] row = new double[rowString.length];
                for (int i = 0; i < rowString.length; i++) {
                    row[i] = Double.parseDouble(rowString[i]);
                }
                dataArray.add(row);
            }

            return dataArray;
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }



}

/*
Part 1: Nearest Neighbour Method [30 Marks for COMP307, and 37 Marks for
        COMP420]
        This part is to implement the k-Nearest Neighbour algorithm, and evaluate the method on the wine data
        set described below. Additional questions on k-means and k-fold cross validation need to be answered and
        discussed.

        Problem Description
        The wine data set is taken from the UCI Machine Learning Repository (http://mlearn.ics.uci.edu/MLRepository.html).
        The data set contains 178 instances in 3 classes, having 59, 71 and 48 instances, respectively. Each instance
        has 13 attributes: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280%2FOD315 of diluted wines, and Proline.
        The data set is split into two subsets, one for training and the other for testing.

        Requirements
        Your program should classify each instance in the test set wine-test according to the training set wine-training.
        Note that the final columns in these files list the class label for each instance.
        Your program should take two file names as command line arguments, and classify each instance in the test
        set (the second file name) according to the training set (the first file name).


        You should submit the following files electronically and also a report in hard copy.

        • (15 marks) Program code for your k-Nearest Neighbour Classifier (both the source code and the
        executable program that is runnable on ECS School machines).
        • readme.txt describing how to run your program, and

        • (15 marks) A report in PDF, text or DOC format. The report should include:

        1
        1. Report the class labels of each instance in the test set predicted by the basic nearest neighbour
        method (where k=1), and the classification accuracy on the test set of the basic nearest neighbour
        method.


        2. Report the classification accuracy on the test set of the k-nearest neighbour method where k=3,
        and compare and comment on the performance of the two classifiers (k=1 and k=3).


        3. Discuss the main advantages and disadvantages of k-Nearest Neighbour method.


        4. Assuming that you are asked to apply the k-fold cross validation method for the above problem
        with k=5, what would you do? State the major steps.


        5. In the above problem, assuming that the class labels are not available in the training set and the
        test set, and that there are three clusters, which method would you use to group the examples in
        the data set? State the major steps.


        • (7 marks, this question is for COMP420 students. It is optional for COMP307 students (bonus 5
        marks))

        Implement the clustering method, run it on the wine data set using the number of clusters of 3 and 5.
        Submit the program code, report the number of instances in each cluster, and provide an analysis on
        your results (you can just provide simple comments by comparing your results with the classes).







        Part 2: Decision Tree Learning Method [35 Marks for COMP307]

        This part involves writing a program that implements a simple version of the Decision Tree (DT) learning
        algorithm, reporting the results, and discussing your findings.

        Problem Description
        The main data set for the DT program is in the files hepatitis, hepatitis-training, and hepatitis-test.
        It describes 137 cases of patients with hepatitis, along with the outcome. Each case is specified by 16 Boolean
        attributes describing the patient and the results of various tests. The goal is to be able to predict the outcome
        based on the attributes. The first file contains all the 137 cases; the training file contains 112 of the cases
        (chosen at random) and the testing file contains the remaining 25 cases. The first columns of the files show
        the class label (“live” or “die”) of each instance. The data files are formatted as tab-separated text files,
        containing one header line, followed by a line for each instance.

        • The first line contains the names of the attributes.
        • Each instance line contains the class name followed by the values of the attributes (“true” or “false”).

        This data set is taken from the UCI Machine Learning Repository http://mlearn.ics.uci.edu/MLRepository.html.
        It consists of data about the prognosis of patients with hepatitis. This version has been simplified by removing
        some of the numerical attributes, and converting others to booleans.
        The file golf.data is a smaller data set in the same format that may be useful for testing your programs
        while you are getting them going. Each instance describes the weather conditions that made a golf player
        decide to play golf or to stay at home. This data set is not large enough to do any useful evaluation.
        Decision Tree Learning Algorithm

        The basic algorithm for building decision trees from examples is relatively easy. Complications arise in
        handling multiple kinds of attributes, doing the statistical significance testing, pruning the tree, etc., but
        your program don’t have to deal with any of the complications.
        For the simplest case of building a decision tree for a set of instances with boolean attributes (yes/no
        decisions), with no pruning, the algorithm is shown below.

     2 instances: the set of training instances that have been provided to the node being constructed;
        attributes: the list of attributes that were not used on the path from the root to this node.
        BuildTree (Set instances, List attributes)
        if instances is empty
        return a leaf node containing the name and probability of the most
        probable class across the whole dataset(i.e., the ‘‘baseline’’ predictor)
        if instances are pure (i.e., all in the same class)
        return a leaf node containing the name of the class of the instances
        in the node and probability 1
        if attributes is empty
        return a leaf node containing the name and probability of the
        majority class of the instances in the node (choose randomly
        if classes are equal)
        else find best attribute:
        for each attribute
        separate instances into two sets:
        1) instances for which the attribute is true, and
        2) instances for which the attribute is false
        compute purity of each set.
        if weighted average purity of these sets is best so far
        bestAtt = this attribute
        bestInstsTrue = set of true instances
        bestInstsFalse = set of false instances
        build subtrees using the remaining attributes:
        left = BuildTree(bestInstsTrue, attributes - bestAtt)
        right = BuildTree(bestInstsFalse, attributes - bestAttr)
        return Node containing (bestAtt, left, right)
        To apply a constructed decision tree to a test instance, the program will work down the decision tree, choosing
        the branch based on the value of the relevant attribute in the instance, until it gets to a leaf. It then returns
        the class name in that leaf.
        Requirements
        Your program should take two file names as command line arguments, construct a classifier from the training
        data in the first file, and then evaluate the classifier on the test data in the second file.
        You can write your program in Java, C/C++, Python, or any other programming language.
        You should submit the following files electronically and also a report in hard copy.
        • (20 marks) Program code for your decision tree classifier (both source code and executable program
        running on the ECS School machines). The program should print out the tree in a human readable
        form (text form is fine).
        You should use the (im)purity measures presented in the lectures unless you want to use another
        measure from the textbook (i.e. information gain), which is more complex and not recommended. The
        file helper-code.java contains java code that may be useful for reading instance data from the data
        files. You may use it if you wish, but you may also write your own code.
        • readme.txt describing how to run your program, and
        • (15 marks) A report in PDF, text or DOC format. The report should include:
        1. You should first apply your program to the hepatitis-training and hepatitis-test files and
        report the classification accuracy in terms of the fraction of the test instances that it classified
        correctly. Report the learned decision tree classifier printed by your program. Compare the
        accuracy of your Decision Tree program to the baseline classifier which always predicts the most
        frequent class in the dataset, and comment on any difference.
        2. You could then construct 10 other pairs of training/test files, train and test your classifiers on each
        pair, and calculate the average accuracy of the classifiers over the 10 trials. The files of 10 split
        10 training and test sets are provided. The files are named as hepatitis-training-run-*, and
        hepatitis-test-run-*. Each training set has 107 instances and each test set has the remaining
        30 instances. Show you working.
        3
        3. “Pruning” (removing) some of leaves of the decision tree will always make the decision tree less
        accurate on the training set. Explain (a) How you could prune leaves from the decision tree; (b)
        Why it would reduce accuracy on the training set, and (c)Why it might improve accuracy on the
        test set.
        4. Explain why the impurity measure is not a good measure if there are three or more classes that
        the decision tree must distinguish.
        • (3 marks, only for COMP420 students) What three conditions are needed if a function (such as
        P(A)*P(B)) is used as an impurity measure in building a decision tree?
        Note: A Simple Way of Outputting a Learned Decision Tree
        The easiest way of outputting the tree is to do a traversal of the tree. For each non-leaf node, print out the
        name of the attribute, then print the left tree, then print the right tree. For each leaf node, print out the
class name in the leaf node and the fraction. By increasing the indentation on each recursive call, it becomes
        somewhat readable.
        Here is a sample tree (not a correct tree for the golf dataset). Note that the final leaf node which is impure
        can only occur on a path which has used all the attributes so there is no attribute left to split the instances
        any further. This should not happen for the data set above, but might happen for some datasets.
        cloudy = True:
        raining = True:
        Class StayHome, prob = 1.0
        raining = False:
        Class PlayGolf, prob = 1.0
        cloudy = False:
        hot = true:
        Class PlayGolf, prob = 1.0
        hot = False:
        windy = True:
        Class StayHome, prob = 1.0
        windy = False:
        Class PlayGolf, prob = 0.75
        Here is some sample (Java) code for outputting a tree.
        In class Node (a non-leaf node of a tree):
public void report(String indent){
        System.out.format("%s%s = True:\n",
        indent, attName);
        left.report(indent+" ");
        System.out.format("%s%s = False:\n",
        indent, attName);
        right.report(indent+" ");
        }
        In class Leaf (a leaf node of a tree):
public void report(String indent){
        if (count==0)
        System.out.format("%sUnknown\n", indent);
        else
        System.out.format("%sClass %s, prob=$4.2f\n",
        indent, className, probability);
        }

        */