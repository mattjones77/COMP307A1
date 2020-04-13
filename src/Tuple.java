
class Tuple<Tuple> {
    private double dist;
    private double label;

    Tuple(double dist, double label) {
        this.dist = dist;
        this.label = label;
    }


    double getDist() {
        return dist;
    }

    double getLabel() {
        return label;
    }

}