package my.ml.impl.tree;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.IntStream;

public class DecisionTree<T> {
    static class Node<T> {
        Node<T> right;
        Node<T> left;

        Predicate<T> predicate;

        boolean predict(T t) {
            return predicate.test(t);
        }

    }

    private Node<T> root;

    public boolean predict(T t) {
        return predictRecursive(root, t);
    }

    private boolean predictRecursive(Node node, T t) {
        if (node.predicate == null) {
            throw new IllegalStateException();
        }

        if (node.predict(t)) {
            if (node.left == null) {
                return true;
            }
            return predictRecursive(node.left, t);
        } else {
            if (node.right == null) {
                return false;
            }
            return predictRecursive(node.right, t);
        }
    }

    public void fit() {
        double[][] matrixData = {{1d, 2d, 3d}, {2d, 5d, 3d}};
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);
    }

    /**
     * explanatory variable -> n * m matrix
     * - search condition which maximize entropy
     * - condition become function object to node
     * target variable -> result of training
     * - immutable
     * - collection
     * to simplify, Let's start categorical input
     */

    public void transform() {

    }

    // test each predicate and pick one which minimize entropy!
    private void selectPredicate(RealMatrix input) {
        // column type is categorical -> predicate is simple
        // column type is numeric -> predicate is complex one, so use logistic regression


    }


    // pick one col
    // test given predicate function
    // return entropy
    private <ColumnType> double test(List<ColumnType> input, List<Boolean> output, Predicate<ColumnType> predicate) {
        // split to two group by given predicate
        var g1 = new ArrayList<Boolean>();
        var g2 = new ArrayList<Boolean>();
        for (int i = 0; i < input.size(); i++) {
            if (predicate.test(input.get(i))) {
                g1.add(output.get(i));
            } else {
                g2.add(output.get(i));
            }
        }
        return Math.min(getEntropy(g1), getEntropy(g2));
    }

    private double getEntropy(List<Boolean> input) {
        int N = input.size();
        int n0 = 0;
        int n1 = 0;
        for (var i : input) {
            if (i) {
                n0++;
            } else {
                n1++;
            }
        }
        return IntStream.of(n0, n1).mapToDouble(i -> {
            var portion = i / N;
            return -1 * portion * Math.log10(portion) / Math.log10(2);
        }).sum();
    }

}
