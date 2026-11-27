package ma.enset.nn.config;

/**
 * Singleton (amelioration) : configuration globale du reseau de neurones.
 */
public final class NetworkConfig {

    private static final NetworkConfig INSTANCE = new NetworkConfig();

    private static final int DEFAULT_EPOCHS = 50;
    private static final double DEFAULT_LEARNING_RATE = 0.1;
    private static final long DEFAULT_SEED = 42L;

    private int epochs = DEFAULT_EPOCHS;
    private double learningRate = DEFAULT_LEARNING_RATE;
    private long randomSeed = DEFAULT_SEED;

    private NetworkConfig() {
    }

    public static NetworkConfig getInstance() {
        return INSTANCE;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Le nombre d'epochs doit etre positif");
        }
        this.epochs = epochs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Le learning rate doit etre dans ]0, 1]");
        }
        this.learningRate = learningRate;
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }

    public void resetDefaults() {
        this.epochs = DEFAULT_EPOCHS;
        this.learningRate = DEFAULT_LEARNING_RATE;
        this.randomSeed = DEFAULT_SEED;
    }
}
