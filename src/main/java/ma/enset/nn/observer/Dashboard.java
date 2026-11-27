package ma.enset.nn.observer;

import ma.enset.nn.state.NetworkStateType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Tableau de bord : observateur concret qui affiche et agrege les statistiques
 * sur les changements d'etat du reseau.
 */
public class Dashboard implements NetworkObserver {

    private final Map<NetworkStateType, Integer> transitionCounts = new EnumMap<>(NetworkStateType.class);
    private final List<NetworkEvent> history = new ArrayList<>();
    private final List<String> printedLines = new ArrayList<>();

    @Override
    public void onNetworkEvent(NetworkEvent event) {
        history.add(event);
        transitionCounts.merge(event.state(), 1, Integer::sum);

        String line = String.format("[Dashboard] %s | etat=%-12s | %s",
                event.networkName(), event.state(), event.message());
        printedLines.add(line);
        System.out.println(line);
    }

    public int count(NetworkStateType type) {
        return transitionCounts.getOrDefault(type, 0);
    }

    public List<NetworkEvent> getHistory() {
        return Collections.unmodifiableList(history);
    }

    public List<String> getPrintedLines() {
        return Collections.unmodifiableList(printedLines);
    }

    public String summary() {
        return String.format("Dashboard{transitions=%s, derniere=%s}",
                transitionCounts, history.isEmpty() ? "aucune" : history.get(history.size() - 1).state());
    }
}
