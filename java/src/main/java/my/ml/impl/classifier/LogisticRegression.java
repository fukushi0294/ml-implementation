package my.ml.impl.classifier;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import java.util.List;

/**
 * This is basic implementation for logistic regression
 * TODO: Assume input is one dimension. because use for DecisionTree
 *      log(p/(1-p)) = b0 + b1 * x1 + b2 * x2^2 + ...
 *     Likely hood
 *         Π(p_i^y_i*(1-p_i)^(1-y_i)) = Y_transpose*X*B - Σ(1 + exp(X*B))
 */
public class LogisticRegression {
    RealMatrix weights;
    List<Boolean> output;

    public boolean predict() {
        return false;
    }

    public void transform() {
//        weights = List.of(0d, 0d, 0d);
    }

    public void sample() {
        UnivariateFunction basicF = new UnivariateFunction() {
            public double value(double x) {
                return x * FastMath.sin(x);
            }
        };

        int params = 1;
        int order = 3;
        double xRealValue = 2.5;
        DerivativeStructure x = new DerivativeStructure(params, order, 0, xRealValue);
        DerivativeStructure y = f(x);
    }

    static class LikelyHood {
        RealMatrix y;
        RealMatrix x;
        RealMatrix b;

        public void train() {

        }

        private double calc(RealMatrix b) {
            var matrixPart = y.transpose().multiply(x).multiply(b).getNorm();
            var t = x.multiply(b);
            var length = t.getRowDimension();
            var scalaPart = 0;
            for (int i = 0; i < length; i++) {
                scalaPart -= (1 + Math.exp(t.getRowMatrix(i).getNorm()));
            }
            return matrixPart + scalaPart;
        }
    }


    private void train(RealMatrix y, RealMatrix x) {
        var likelyHood = y.transpose().multiply(x).multiply(weights);


    }


}
