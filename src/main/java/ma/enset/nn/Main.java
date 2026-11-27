package ma.enset.nn;

import ma.enset.nn.factory.ActivationFactory;
import ma.enset.nn.aspect.SecurityContext;
import ma.enset.nn.builder.NeuralNetworkBuilder;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.SimpleLayer;
import ma.enset.nn.decorator.BatchNormalizationDecorator;
import ma.enset.nn.decorator.DropoutDecorator;
import ma.enset.nn.observer.Dashboard;

import java.util.Random;

/**
 * Demonstration executable de l'examen blanc :
 * construction du reseau 3-5-2, etats, observateur, decorators, securite et AOP.
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) {
        System.out.println("==============================================================");
        System.out.println(" Examen Blanc Design Pattern - Reseau de neurones (GLSID)");
        System.out.println("==============================================================");

        NetworkConfig config = NetworkConfig.getInstance();
        config.setEpochs(30);
        config.setLearningRate(0.1);

        Dashboard dashboard = new Dashboard();

        Layer hiddenLayer = new DropoutDecorator(
                new BatchNormalizationDecorator(
                        new SimpleLayer(LayerType.HIDDEN, 5,
                                ActivationFactory.create("sigmoid"), new Random(42))),
                0.2, new Random(7));

        NeuralNetwork network = new NeuralNetworkBuilder()
                .name("reseau-3-5-2")
                .seed(42)
                .inputLayer(3, ActivationFactory.create("identity"))
                .layer(hiddenLayer)
                .outputLayer(2, ActivationFactory.create("softmax"))
                .observer(dashboard)
                .build();

        System.out.println("\n[1] Architecture du reseau");
        for (Layer layer : network.getLayers()) {
            System.out.println("    - " + layer.description());
        }

        System.out.println("\n[2] Etat initial : " + network.getStateName());
        try {
            network.predict(new double[]{0.1, 0.2, 0.3});
        } catch (IllegalStateException exception) {
            System.out.println("    predict() refuse : " + exception.getMessage());
        }

        System.out.println("\n[3] Securite : tentative d'entrainement avec le role GUEST");
        SecurityContext.setRole(SecurityContext.GUEST_ROLE);
        try {
            network.train();
        } catch (SecurityException exception) {
            System.out.println("    " + exception.getMessage());
        }
        SecurityContext.setRole(SecurityContext.DATA_SCIENTIST_ROLE);

        System.out.println("\n[4] Entrainement via fit()");
        network.fit();
        System.out.println("    Etat final : " + network.getStateName());
        System.out.println("    Epochs effectues : " + network.getLossHistory().size());
        double[] prediction = network.predict(new double[]{0.2, 0.5, 0.9});
        System.out.printf("    Prediction : [%.4f, %.4f]%n", prediction[0], prediction[1]);
        System.out.println("    " + dashboard.summary());

        System.out.println("\n[5] Fonctions d'activation : sigmoid, relu, tanh, identity, softmax");
        System.out.println("    + LegacyArcTan adaptee par LegacyActivationAdapter");

        System.out.println("\n[6] Journalisation AOP : logs/neural-network.log");
        System.out.println("==============================================================");
        System.out.println(" Fin de la demonstration");
        System.out.println("==============================================================");
    }
}
