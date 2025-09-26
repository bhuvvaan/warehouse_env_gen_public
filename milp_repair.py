import numpy as np
import pulp
from collections import defaultdict

def _nbr4(r, c, H, W):
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W:
            yield (rr, cc)

def repair_layout_milp(
    layout_arr: np.ndarray,
    Ns: int,                          # total shelves after repair
    source_coord=None,                # (r,c) empty cell for dummy source; auto-chosen if None
    require_endpoint_adj: bool=True,  # every endpoint touches >=1 shelf
    require_shelf_two_endpoints: bool=True, # every shelf touches >=2 endpoints
    solver_name="CBC"
):
    """
    MILP repair for layouts with values:
      1.0 = shelf, -1.0 = endpoint, 0.0 = empty

    Constraints:
      * exactly one tile per cell
      * fixed number of shelves (Ns)
      * endpoint adjacency: each endpoint has >=1 adjacent shelf
      * shelf adjacency: each shelf has >=2 adjacent endpoints (optional)
      * all non-shelf cells reachable from a dummy source via flow
        (shelves block flow)

    Returns: np.ndarray of same shape, values in {1.0, -1.0, 0.0}
    """
    if layout_arr.ndim != 2:
        raise ValueError("layout_arr must be 2D")
    H, W = layout_arr.shape

    allowed = {1.0, -1.0, 0.0}
    vals = set(np.unique(layout_arr).tolist())
    if not vals.issubset(allowed):
        raise ValueError(f"Unexpected values {vals - allowed}; allowed {allowed}")

    V = [(r,c) for r in range(H) for c in range(W)]
    E = [((r,c),nbr) for r,c in V for nbr in _nbr4(r,c,H,W)]

    # Pick source
    if source_coord is None:
        candidates = [(r,c) for r,c in V if layout_arr[r,c] == 0.0]
        if not candidates:
            raise ValueError("Need at least one empty cell for source")
        source_coord = candidates[0]
    src = tuple(map(int, source_coord))

    # Tile types
    TYPES = ("p","e","s")  # empty, endpoint, shelf

    # One-hot from original
    def x0(v,t):
        r,c=v
        val=layout_arr[r,c]
        if t=="s": return 1 if val==1.0 else 0
        if t=="e": return 1 if val==-1.0 else 0
        if t=="p": return 1 if val==0.0 else 0
        return 0

    prob = pulp.LpProblem("RepairLayout", pulp.LpMinimize)

    X={(v,t):pulp.LpVariable(f"x_{v[0]}_{v[1]}_{t}",cat="Binary") for v in V for t in TYPES}
    D={v:pulp.LpVariable(f"d_{v[0]}_{v[1]}",cat="Binary") for v in V}
    BIG=len(V)
    F={(u,v):pulp.LpVariable(f"f_{u[0]}_{u[1]}__{v[0]}_{v[1]}",lowBound=0) for (u,v) in E}
    Fs={v:pulp.LpVariable(f"fs_{v[0]}_{v[1]}",lowBound=0) for v in V}
    Ft={v:pulp.LpVariable(f"ft_{v[0]}_{v[1]}",lowBound=0) for v in V}

    # Objective = Hamming distance
    costs={(v,t):(0 if x0(v,t)==1 else 1) for v in V for t in TYPES}
    prob += pulp.lpSum(costs[(v,t)]*X[(v,t)] for v in V for t in TYPES)

    # Exactly one tile + unique dummy
    for v in V:
        prob += pulp.lpSum(X[(v,t)] for t in TYPES) + D[v] == 1
        if v==src: prob += D[v]==1
        else:      prob += D[v]==0

    # Fixed shelves
    prob += pulp.lpSum(X[(v,"s")] for v in V) == Ns

    # Adjacency
    if require_endpoint_adj:
        for v in V:
            nbrs=list(_nbr4(v[0],v[1],H,W))
            prob += pulp.lpSum(X[(u,"s")] for u in nbrs) >= X[(v,"e")]

    if require_shelf_two_endpoints:
        for v in V:
            nbrs=list(_nbr4(v[0],v[1],H,W))
            prob += 2*X[(v,"s")] <= pulp.lpSum(X[(u,"e")] for u in nbrs)

    # Reachability
    for v in V:
        prob += Ft[v]==X[(v,"p")]+X[(v,"e")]
    for v in V:
        prob += Fs[v]<=BIG*D[v]

    in_sum, out_sum = defaultdict(list), defaultdict(list)
    for (u,v) in E:
        out_sum[u].append(F[(u,v)])
        in_sum[v].append(F[(u,v)])
    for v in V:
        prob += Fs[v]+pulp.lpSum(in_sum[v]) == Ft[v]+pulp.lpSum(out_sum[v])
    for (u,v) in E:
        prob += F[(u,v)] <= BIG*(1-X[(u,"s")])

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300) if solver_name.upper()=="CBC" else pulp.CPLEX_PY(msg=True)
    status = prob.solve(solver)
    if pulp.LpStatus[status]!="Optimal":
        raise RuntimeError(f"Solve status: {pulp.LpStatus[status]}")

    # Extract
    repaired=np.zeros((H,W),dtype=np.float32)
    for r,c in V:
        v=(r,c)
        if D[v].value()>=0.5: repaired[r,c]=0.0
        elif X[(v,"s")].value()>=0.5: repaired[r,c]=1.0
        elif X[(v,"e")].value()>=0.5: repaired[r,c]=-1.0
        else: repaired[r,c]=0.0
    return repaired
