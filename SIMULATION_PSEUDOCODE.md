# Tactical Drone Network Simulation Algorithm

This simulator evaluates the performance of the optimized drone delivery network using real-world waybill data from Meituan.

## Algorithm Pseudocode

**Input:** 
- $\mathcal{O}$: Set of orders (timestamp, pickup location $i$, delivery location $j$)
- $\mathcal{Y}$: Set of optimized facility locations (Bases $L$, Pickups $P$, Deliveries $D$, Swapping $E$)
- $N_{drone}$: Number of drones per physical base
- $ChainEnabled$: Boolean flag for consecutive delivery logic

**Function SIMULATE($\mathcal{O}, \mathcal{Y}$):**
1.  **Initialize:**
    - $E \leftarrow$ Priority queue of all orders in $\mathcal{O}$ as `NewOrder` events
    - $A \leftarrow$ Set of available drones, initialized with $N_{drone}$ at each physical base $L \in \hat{\mathcal{L}}$
    - $Q \leftarrow \emptyset$ (Queue for waiting orders)
2.  **While** $E$ is not empty:
    - $e \leftarrow E.pop\_next()$
    - $t_c \leftarrow e.time$
    - **If** $e$ is `NewOrder`:
        - Search for closest available drone $d \in A$
        - **If** $ChainEnabled$ and $d$ is at a virtual base:
            - Check proximity to order pickup $i$
        - **Else if** $d$ is at its assigned physical base:
            - Check assignment map ($y_{li} > 0$)
        - **If** drone $d$ found:
            - $d.status \leftarrow Busy$
            - $A \leftarrow A \setminus \{d\}$
            - $t_{travel} \leftarrow Dist(d.pos, i) / Speed$
            - Schedule `AtPickup` event at $t_c + t_{travel}$
        - **Else**:
            - $Reject$ or $Queue$ the order
    - **Else if** $e$ is `AtPickup`:
        - $t_{service} \leftarrow \text{On-scene time}$
        - $t_{travel} \leftarrow Dist(i, j) / Speed$
        - Schedule `AtDelivery` event at $t_c + t_{service} + t_{travel}$
    - **Else if** $e$ is `AtDelivery`:
        - **If** $ChainEnabled$:
            - $d.pos \leftarrow j$
            - $d.status \leftarrow Available$ (Drone stays as virtual base)
            - $A \leftarrow A \cup \{d\}$
        - **Else**:
            - $t_{return} \leftarrow Dist(j, \text{HomeBase}) / Speed$
            - Schedule `DroneAvailable` event at $t_c + t_{return}$
    - **Else if** $e$ is `DroneAvailable`:
        - $d.pos \leftarrow \text{HomeBase}$
        - $d.status \leftarrow Available$
        - $A \leftarrow A \cup \{d\}$
        - **If** $Q$ not empty:
            - Attempt to assign $d$ to next order in $Q$
3.  **Output:** 
    - Service Rate, Average Flight Time, Total Distance, Congestion Metrics.
