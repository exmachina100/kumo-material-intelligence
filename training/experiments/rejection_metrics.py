"""Pure-Python confidence rejection diagnostics for research models.

These are diagnostic coverage/false-acceptance measures, not calibrated
confidence estimates or a substitute for independent out-of-domain testing.
"""

def rejection_summary(in_domain_scores, out_domain_scores, thresholds=(0.5,0.65,0.8,0.9)):
    inside=[float(x) for x in in_domain_scores]
    outside=[float(x) for x in out_domain_scores]
    for value in inside+outside:
        if not (0 <= value <= 1):
            raise ValueError("Expected finite scores in [0, 1]")
    rows=[]
    for threshold in thresholds:
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold outside [0,1]")
        in_accepted=sum(score >= threshold for score in inside)
        out_accepted=sum(score >= threshold for score in outside)
        rows.append({
            "threshold":threshold,
            "in_domain_accept_rate":round(in_accepted/len(inside),4) if inside else None,
            "out_domain_false_accept_rate":round(out_accepted/len(outside),4) if outside else None,
            "in_accepted":in_accepted,"out_accepted":out_accepted
        })
    return {
        "in_domain_count":len(inside),"out_domain_count":len(outside),
        "in_mean_max_softmax":round(sum(inside)/len(inside),4) if inside else None,
        "out_mean_max_softmax":round(sum(outside)/len(outside),4) if outside else None,
        "thresholds":rows,
        "warning":"Softmax is not calibrated. Proxy-only diagnostics; NOT evidence of safe unknown rejection in production."
    }
