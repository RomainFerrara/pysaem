"""Data container: observations, events, regressors and covariates per subject.

Arrays are flat and indexed by pointer arrays (subject i owns ptr[i]:ptr[i+1]),
the layout the compiled kernels read.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


class DataError(ValueError):
    """The data cannot be used; the message says why."""


_BLANK = (".", "", "NA", "NaN", "nan")  # text that stands for a missing value


@dataclass
class Dataset:
    ids: np.ndarray
    output_labels: list
    obs_ptr: np.ndarray
    obs_time: np.ndarray
    obs_y: np.ndarray
    obs_out: np.ndarray
    obs_cens: np.ndarray
    obs_limit: np.ndarray
    obs_grid: np.ndarray
    grid_ptr: np.ndarray
    grid_time: np.ndarray
    ev_ptr: np.ndarray
    ev_time: np.ndarray
    ev_input: np.ndarray
    ev_amt: np.ndarray
    ev_rate: np.ndarray
    ev_reset: np.ndarray            # 1: every state goes back to its initial value first
    ev_ss: np.ndarray               # interval of a steady state reached before the event, else 0
    occ_ptr: np.ndarray             # occasions of subject i: rows occ_ptr[i] to occ_ptr[i + 1]
    occ_time: np.ndarray            # start of each occasion, -inf for the first of a subject
    reg_ptr: np.ndarray
    reg_time: np.ndarray
    reg_value: np.ndarray           # blank cells carry the previous value forward (first backward)
    regressor_names: list
    covariates: dict
    t_first: np.ndarray
    dropped_ids: list = field(default_factory=list)
    occ_labels: list = field(default_factory=list)  # label of each occasion row
    covariate_levels: dict = field(default_factory=dict)  # categorical: labels in code order
    covariate_groups: dict = field(default_factory=dict)  # categorical with more than two levels:
    # the indicator columns of its levels after the first, which is the reference
    obs_input: np.ndarray = None    # observation whose row is also an input; None: no such row
    reg_given: np.ndarray = None    # cells of reg_value that hold a record; None: all
    input_labels: list = field(default_factory=list)  # label of each input number
    order_given: tuple = (False, False)  # output_labels, input_labels given by the caller

    def __post_init__(self):
        if self.obs_input is None:
            self.obs_input = np.zeros(len(self.obs_y), bool)
        if self.reg_given is None:
            self.reg_given = np.ones(np.shape(self.reg_value), bool)

    n_subjects = property(lambda self: len(self.ids))
    n_obs = property(lambda self: len(self.obs_y))
    n_outputs = property(lambda self: len(self.output_labels))
    n_events = property(lambda self: len(self.ev_time))

    @property
    def n_obs_by_output(self) -> np.ndarray:
        return np.bincount(self.obs_out, minlength=self.n_outputs)

    @property
    def obs_subject(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_subjects), np.diff(self.obs_ptr))

    def matched_to(self, outputs, inputs) -> "Dataset":
        """The data numbered as the outputs and inputs of a model. A label that names an output
        or an input goes to it. Other labels keep their order: that of output_labels or
        input_labels when given, numerical for numbers. Text labels that name nothing are
        refused when the model has more than one output or input."""
        out = _numbering(self.output_labels, list(outputs), self.order_given[0], "output")
        inp = _numbering(self.input_labels, list(inputs), self.order_given[1], "input")
        new = self
        if out is not None:
            new = replace(new, obs_out=out[new.obs_out],
                          output_labels=list(outputs)[:int(out.max()) + 1])
        if inp is not None:
            new = replace(new, ev_input=inp[new.ev_input],
                          input_labels=list(inputs)[:int(inp.max()) + 1])
        return new

    @classmethod
    def from_records(cls, df, *, id="ID", time="TIME", value="DV", amount="AMT", rate="RATE",
                     duration="TINF", event="EVID", missing="MDV", output=None, input=None,
                     cens="CENS", limit="LIMIT", additional="ADDL", interval="II",
                     steady_state="SS", occasion="OCC",
                     covariates=(), regressors=(), output_labels=None,
                     input_labels=None) -> "Dataset":
        """One row per record, observations and events mixed (the usual event-record layout).

        A row is an event, an input or a reset, when its event flag is 1, 3 or 4 (3 resets
        the states, 4 resets them and gives the amount), or, without flag column, when its
        amount is present and non-zero. Without flag column, an event row with a value not
        marked missing is an observation too; obs_input marks it. An event with n additional
        occurrences is repeated n times, interval apart. Optional columns may be absent.
        """
        col = _resolver(df)
        need = {k: col(v, required=True) for k, v in (("id", id), ("time", time),
                                                      ("value", value))}
        t = _numeric(df[need["time"]], need["time"])
        y = _numeric(df[need["value"]], need["value"])
        amt_c, evid_c, mdv_c = col(amount), col(event), col(missing)
        amt = _numeric(df[amt_c], amt_c) if amt_c else pd.Series(np.nan, index=df.index)
        if evid_c:
            evid = _numeric(df[evid_c], evid_c).fillna(0)
            # 3 resets the states, 4 resets them and gives the amount
            is_event = evid.isin([1, 3, 4])
            is_obs = (evid == 0) & y.notna()
        else:
            evid = None
            is_event = amt.notna() & (amt != 0)
            is_obs = y.notna()
        if mdv_c:
            is_obs &= _numeric(df[mdv_c], mdv_c).fillna(0) == 0

        obs = pd.DataFrame({"id": df[need["id"]], "time": t, "value": y})[is_obs]
        obs["obs_input"] = is_event[is_obs]
        occ_c = col(occasion)
        if occ_c:
            obs["occasion"] = df.loc[is_obs, occ_c]
        for key, name in (("output", output), ("cens", cens), ("limit", limit)):
            c = col(name)
            if c:
                obs[key] = df.loc[is_obs, c]
        events = None
        if is_event.any():
            events = pd.DataFrame({"id": df[need["id"]], "time": t, "amount": amt})[is_event]
            if occ_c:
                events["occasion"] = df.loc[is_event, occ_c]
            if evid is not None:
                events["reset"] = evid[is_event].isin([3, 4]).astype(int)
                events.loc[evid[is_event] == 3, "amount"] = 0.0
            for key, name in (("rate", rate), ("duration", duration), ("input", input)):
                c = col(name)
                if c:
                    events[key] = df.loc[is_event, c]
            if evid is not None:
                # a reset without amount gives no input: its rate or duration is ignored
                for key in ("rate", "duration"):
                    if key in events.columns:
                        events.loc[(evid[is_event] == 3).to_numpy(), key] = np.nan
            add_c, ii_c, ss_c = col(additional), col(interval), col(steady_state)
            if ss_c:
                ss = _numeric(df.loc[is_event, ss_c], ss_c).fillna(0)
                if (~ss.isin([0, 1])).any():
                    raise DataError("steady state: only SS = 0 or 1 is supported")
                ii = _numeric(df.loc[is_event, ii_c], ii_c) if ii_c else pd.Series(np.nan, index=ss.index)
                if (ss.eq(1) & ~(ii > 0)).any():
                    raise DataError("a steady-state record needs a positive interval II")
                # the record says the subject is at steady state of that input every II
                # before it; the states are reset first (the usual meaning of SS = 1)
                events["ss"] = np.where(ss.eq(1), ii, 0.0)
                events.loc[ss.eq(1).to_numpy(), "reset"] = 1
            if add_c:
                events = _repeat(events, _numeric(df.loc[is_event, add_c], add_c),
                                 _numeric(df.loc[is_event, ii_c], ii_c) if ii_c else None)
        regs = None
        if regressors:
            cols = [col(r, required=True) for r in regressors]
            regs = pd.DataFrame({"id": df[need["id"]], "time": t})
            for name, c in zip(regressors, cols):
                regs[name] = _numeric(df[c], c)
        covs = None
        if covariates:
            cols = [col(c, required=True) for c in covariates]
            covs = df[[need["id"], *cols]].rename(
                columns=dict(zip([need["id"], *cols], ["id", *covariates])))
        return cls.from_tables(obs, events=events, regressors=regs, covariates=covs,
                               output_labels=output_labels, input_labels=input_labels)

    @classmethod
    def from_tables(cls, observations, *, events=None, regressors=None, covariates=None,
                    output_labels=None, input_labels=None) -> "Dataset":
        """Observations (id, time, value[, output, cens, limit, occasion, obs_input]) and optional
        tables: events (id, time, amount[, rate, duration, input, reset, ss, occasion]),
        regressors (id, time, one column per regressor), covariates (id, one column per
        covariate). obs_input marks an observation whose row is also an input (from_records).

        Occasions follow one another in time within a subject; each starts at its first record.
        """
        obs = observations.copy()
        for c in ("id", "time", "value"):
            if c not in obs.columns:
                raise DataError(f"observations: column {c!r} is missing")
        obs["time"] = _numeric(obs["time"], "time")
        obs["value"] = _numeric(obs["value"], "value")
        obs = obs[obs["value"].notna()]
        if obs.empty:
            raise DataError("the data contain no observation")
        if obs["time"].isna().any():
            raise DataError("observations: missing time")

        if "output" in obs.columns:
            labels_seen = obs["output"].map(_label)
            labels = [str(x) for x in output_labels] if output_labels else sorted(
                labels_seen.unique(), key=_label_key)
            unknown = sorted(set(labels_seen) - set(labels))
            if unknown:
                raise DataError(f"observation type(s) {unknown} not in {labels}")
            obs["out"] = labels_seen.map({lab: i for i, lab in enumerate(labels)})
        else:
            labels = [str(x) for x in output_labels] if output_labels else ["y"]
            obs["out"] = 0

        all_ids = list(pd.unique(pd.concat(
            [t["id"] for t in (observations, events, regressors, covariates) if t is not None])))
        ids = [i for i in pd.unique(obs["id"])]
        dropped = [i for i in all_ids if i not in set(ids)]
        index = {v: k for k, v in enumerate(ids)}
        n = len(ids)

        obs["subj"] = obs["id"].map(index)
        obs = obs.sort_values(["subj", "time", "out"], kind="stable")
        subj = obs["subj"].to_numpy()
        obs_ptr = _pointers(subj, n)
        tt = obs["time"].to_numpy(float)
        new_point = np.ones(len(obs), bool)
        new_point[1:] = (subj[1:] != subj[:-1]) | (tt[1:] != tt[:-1])
        obs_grid = np.cumsum(new_point) - 1
        grid_time = tt[new_point]
        grid_ptr = _pointers(subj[new_point], n)
        cens = _numeric(obs["cens"], "cens").fillna(0).to_numpy(int) \
            if "cens" in obs.columns else np.zeros(len(obs), int)
        limit = _numeric(obs["limit"], "limit").to_numpy(float) \
            if "limit" in obs.columns else np.full(len(obs), np.nan)
        obs_input = _numeric(obs["obs_input"], "obs_input").fillna(0).to_numpy() != 0 \
            if "obs_input" in obs.columns else np.zeros(len(obs), bool)

        if events is not None and len(events):
            ev = events.copy()
            ev = ev[ev["id"].isin(index)]
            ev["subj"] = ev["id"].map(index)
            ev["time"] = _numeric(ev["time"], "event time")
            ev["amount"] = _numeric(ev["amount"], "amount")
            if ev[["time", "amount"]].isna().any().any():
                raise DataError("events: missing time or amount")
            r = _numeric(ev["rate"], "rate").fillna(0).to_numpy(float) \
                if "rate" in ev.columns else np.zeros(len(ev))
            if "duration" in ev.columns:
                dur = _numeric(ev["duration"], "duration").fillna(0).to_numpy(float)
                r = np.where(dur > 0, ev["amount"].to_numpy(float) / np.where(dur > 0, dur, 1), r)
            if (r < 0).any():
                raise DataError("events: negative rate")
            ev["rate"] = r
            ev["reset"] = ev["reset"].fillna(0).astype(int) if "reset" in ev.columns else 0
            ev["ss"] = _numeric(ev["ss"], "ss").fillna(0.0) if "ss" in ev.columns else 0.0
            ss_ii = ev["ss"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                longer = (ss_ii > 0) & (r > 0) & (ev["amount"].to_numpy(float) / r > ss_ii)
            if longer.any():
                raise DataError(f"steady state: an infusion longer than its interval II is not "
                                f"supported (subject {ev['id'][longer].iloc[0]})")
            if "input" in ev.columns:
                seen = ev["input"].map(_label)
                in_labels = [str(x) for x in input_labels] if input_labels else sorted(
                    seen.unique(), key=_label_key)
                unknown = sorted(set(seen) - set(in_labels))
                if unknown:
                    raise DataError(f"input(s) {unknown} not in {in_labels}")
                ev["inp"] = seen.map({lab: i for i, lab in enumerate(in_labels)})
            else:
                in_labels = []
                ev["inp"] = 0
            ev = ev.sort_values(["subj", "time"], kind="stable")
            ev_ptr = _pointers(ev["subj"].to_numpy(), n)
            ev_time = ev["time"].to_numpy(float)
            ev_input = ev["inp"].to_numpy(np.int64)
            ev_amt = ev["amount"].to_numpy(float)
            ev_rate = ev["rate"].to_numpy(float)
            ev_reset = ev["reset"].to_numpy(np.int64)
            ev_ss = ev["ss"].to_numpy(float)
        else:
            in_labels = []
            ev_ptr = np.zeros(n + 1, np.int64)
            ev_time = ev_amt = ev_rate = ev_ss = np.zeros(0)
            ev_input = ev_reset = np.zeros(0, np.int64)

        reg_names = []
        if regressors is not None:
            rg = regressors[regressors["id"].isin(index)].copy()
            reg_names = [c for c in rg.columns if c not in ("id", "time")]
            rg["subj"] = rg["id"].map(index)
            rg["time"] = _numeric(rg["time"], "regressor time")
            rg = rg.sort_values(["subj", "time"], kind="stable")
            reg_given = rg[reg_names].notna().to_numpy()
            rg[reg_names] = rg.groupby("subj")[reg_names].transform(lambda s: s.ffill().bfill())
            if rg[reg_names].isna().any().any():
                raise DataError(f"regressor(s) {reg_names}: a subject has no value")
            reg_ptr = _pointers(rg["subj"].to_numpy(), n)
            reg_time = rg["time"].to_numpy(float)
            reg_value = rg[reg_names].to_numpy(float)
        else:
            reg_ptr = np.zeros(n + 1, np.int64)
            reg_time = np.zeros(0)
            reg_value = np.zeros((0, 0))
            reg_given = np.zeros((0, 0), bool)

        cov, levels, groups = {}, {}, {}
        if covariates is not None:
            cv = covariates[covariates["id"].isin(index)]
            for name in [c for c in cv.columns if c != "id"]:
                if pd.api.types.is_numeric_dtype(cv[name]):
                    vals = cv[name].astype(float)
                else:
                    text = cv[name].astype(str).str.strip()
                    missing = cv[name].isna() | text.isin(_BLANK)
                    if missing.any():
                        raise DataError(f"covariate {name} is missing for a subject")
                    numbers = pd.to_numeric(text, errors="coerce")
                    if numbers.notna().all():
                        vals = numbers.astype(float)
                    elif numbers.notna().any():
                        raise DataError(f"covariate {name} holds the text "
                                        f"{sorted(set(text[numbers.isna()]))[:5]} among numbers")
                    else:
                        # a categorical covariate becomes its code: 0, 1, ... in sorted
                        # label order
                        levels[name] = sorted(text.unique())
                        vals = text.map({lab: k for k, lab in enumerate(levels[name])}).astype(
                            float)
                per = vals.groupby(cv["id"].map(index).to_numpy())
                if (per.nunique(dropna=False) > 1).any():
                    raise DataError(f"covariate {name} varies within a subject")
                first = per.first().reindex(range(n))
                if first.isna().any():
                    raise DataError(f"covariate {name} is missing for a subject")
                if not np.isfinite(first.to_numpy(float)).all():
                    bad = shown([ids[i] for i in np.flatnonzero(~np.isfinite(first.to_numpy(float)))][:5])
                    raise DataError(f"covariate {name} is infinite for subject(s) {bad}")
                column = first.to_numpy(float)
                if len(levels.get(name, ())) > 2:
                    # one indicator per level after the first: its coefficient is the shift of
                    # that level from the reference level
                    groups[name] = [f"{name}_{lab}" for lab in levels[name][1:]]
                    for k, col in enumerate(groups[name], 1):
                        cov[col] = (column == k).astype(float)
                else:
                    cov[name] = column

        t_first = np.array([grid_time[grid_ptr[i]] for i in range(n)])
        for i in range(n):
            if ev_ptr[i + 1] > ev_ptr[i]:
                t_first[i] = min(t_first[i], ev_time[ev_ptr[i]])
        occ_ptr, occ_time, occ_labels = _occasions(obs, events, index, n, ids)
        return cls(
            ids=np.array(ids, dtype=object), output_labels=labels,
            obs_ptr=obs_ptr, obs_time=tt, obs_y=obs["value"].to_numpy(float),
            obs_out=obs["out"].to_numpy(np.int64), obs_cens=cens.astype(np.int64),
            obs_limit=limit, obs_grid=obs_grid.astype(np.int64), grid_ptr=grid_ptr,
            grid_time=grid_time, ev_ptr=ev_ptr, ev_time=ev_time, ev_input=ev_input,
            ev_amt=ev_amt, ev_rate=ev_rate, ev_reset=ev_reset, ev_ss=ev_ss, occ_ptr=occ_ptr,
            occ_time=occ_time, reg_ptr=reg_ptr, reg_time=reg_time,
            reg_value=reg_value, regressor_names=reg_names, covariates=cov,
            t_first=t_first, dropped_ids=dropped, covariate_levels=levels, occ_labels=occ_labels,
            covariate_groups=groups,
            obs_input=obs_input, reg_given=reg_given, input_labels=in_labels,
            order_given=(bool(output_labels), bool(input_labels)))

    def take(self, subjects) -> "Dataset":
        """The subjects at these indices, in this order, repeats allowed, as a bootstrap sample
        needs them: the k-th copy of a subject keeps its records under the id "<id>.<k>"."""
        subjects = np.asarray(subjects, np.int64)

        def rows(ptr, *arrays):
            # the records of the subjects taken, with their new pointers
            ptr = np.asarray(ptr)
            idx = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in subjects]
                                 + [np.zeros(0, np.int64)]).astype(np.int64)
            new_ptr = np.concatenate([[0], np.cumsum(ptr[subjects + 1] - ptr[subjects])])
            return (new_ptr.astype(np.int64), *(a[idx] for a in arrays))

        obs_ptr, obs_time, obs_y, obs_out, obs_cens, obs_limit, obs_grid, obs_input = rows(
            self.obs_ptr, self.obs_time, self.obs_y, self.obs_out, self.obs_cens, self.obs_limit,
            self.obs_grid, self.obs_input)
        grid_ptr, grid_time = rows(self.grid_ptr, self.grid_time)
        # a grid index is absolute: it moves with its subject's grid
        shift = np.repeat(grid_ptr[:-1] - self.grid_ptr[subjects], np.diff(obs_ptr))
        obs_grid = obs_grid + shift
        ev_ptr, ev_time, ev_input, ev_amt, ev_rate, ev_reset, ev_ss = rows(
            self.ev_ptr, self.ev_time, self.ev_input, self.ev_amt, self.ev_rate, self.ev_reset,
            self.ev_ss)
        occ_ptr, occ_time, occ_idx = rows(self.occ_ptr, self.occ_time, np.arange(len(self.occ_time)))
        reg_ptr, reg_time, reg_value, reg_given = rows(self.reg_ptr, self.reg_time, self.reg_value,
                                                       self.reg_given)
        seen, ids = {}, []
        for i in subjects:
            seen[i] = seen.get(i, 0) + 1
            ids.append(self.ids[i] if seen[i] == 1 else f"{self.ids[i]}.{seen[i]}")
        return replace(
            self, ids=np.array(ids, dtype=object), obs_ptr=obs_ptr, obs_time=obs_time, obs_y=obs_y,
            obs_out=obs_out, obs_cens=obs_cens, obs_limit=obs_limit, obs_grid=obs_grid,
            obs_input=obs_input, grid_ptr=grid_ptr, grid_time=grid_time, ev_ptr=ev_ptr,
            ev_time=ev_time, ev_input=ev_input, ev_amt=ev_amt, ev_rate=ev_rate, ev_reset=ev_reset,
            ev_ss=ev_ss, occ_ptr=occ_ptr, occ_time=occ_time,
            occ_labels=[self.occ_labels[k] for k in occ_idx], reg_ptr=reg_ptr, reg_time=reg_time,
            reg_value=reg_value, reg_given=reg_given,
            covariates={k: np.asarray(v)[subjects] for k, v in self.covariates.items()},
            t_first=self.t_first[subjects], dropped_ids=[])

    def on_grid(self, times) -> "Dataset":
        """The same subjects, events, regressors and covariates, observed at times for
        every output: the design used to predict between or beyond the observations. A
        subject's solve starts at the first of times when that comes before its own start,
        which moves its states (at_times keeps the start)."""
        t = np.unique(np.asarray(times, float))
        grid = self.at_times([t] * self.n_subjects)
        return replace(grid, t_first=np.minimum(self.t_first, t[0]))

    def at_times(self, times) -> "Dataset":
        """The same subjects, events, regressors and covariates, every output observed at
        times[i] for subject i. Each subject keeps its t_first: its solve starts where it does
        on its own records, so the states do not move; a time before that start reads the
        states the solve starts from."""
        own = [np.unique(np.asarray(t, float)) for t in times]
        if len(own) != self.n_subjects:
            raise ValueError(f"times are given for {len(own)} subject(s), the data have "
                             f"{self.n_subjects}")
        m = self.n_outputs
        grid_time = np.concatenate(own)
        grid_ptr = np.concatenate([[0], np.cumsum([len(t) for t in own])]).astype(np.int64)
        k = len(grid_time)
        return replace(
            self, obs_ptr=grid_ptr * m, obs_time=np.repeat(grid_time, m),
            obs_y=np.full(k * m, np.nan), obs_out=np.tile(np.arange(m), k),
            obs_cens=np.zeros(k * m, np.int64), obs_limit=np.full(k * m, np.nan),
            obs_grid=np.repeat(np.arange(k), m), grid_ptr=grid_ptr, grid_time=grid_time,
            obs_input=np.zeros(k * m, bool))

    def observations(self) -> pd.DataFrame:
        """The observation records as a table, in kernel order."""
        return pd.DataFrame({
            "id": self.ids[self.obs_subject],
            "time": self.obs_time,
            "output": np.array(self.output_labels, dtype=object)[self.obs_out],
            "value": self.obs_y,
            "cens": self.obs_cens,
        })


def _resolver(df):
    lower = {str(c).lower(): c for c in df.columns}

    def col(name, required=False):
        if name is None:
            return None
        found = name if name in df.columns else lower.get(str(name).lower())
        if found is None and required:
            raise DataError(f"column {name!r} is missing (columns: {list(df.columns)})")
        return found
    return col


def _occasions(obs, events, index, n, ids):
    """Occasion rows of each subject: start times in order, -inf for the first."""
    tables = [x[["id", "time", "occasion"]] for x in (obs, events)
              if x is not None and "occasion" in x.columns]
    if not tables:
        return np.arange(n + 1, dtype=np.int64), np.full(n, -np.inf), [None] * n
    rec = pd.concat(tables)
    rec = rec[rec["id"].isin(index)]
    if rec["occasion"].isna().any():
        raise DataError("occasion: a record has no occasion")
    rec = rec.assign(subj=rec["id"].map(index), label=rec["occasion"].map(_label))
    ptr, times, labels = [0], [], []
    for i in range(n):
        span = rec[rec["subj"] == i].groupby("label")["time"].agg(["min", "max"]).sort_values("min")
        if span.empty:
            span = pd.DataFrame({"min": [-np.inf], "max": [-np.inf]}, index=[None])
        if (span["min"].to_numpy()[1:] <= span["max"].to_numpy()[:-1]).any():
            raise DataError(f"subject {ids[i]}: occasions overlap in time; put them one after "
                            "another (a reset, EVID 4, can start each one afresh)")
        start = span["min"].to_numpy(float, copy=True)
        start[0] = -np.inf
        times.extend(start)
        labels.extend(span.index)
        ptr.append(ptr[-1] + len(span))
    return np.array(ptr, np.int64), np.array(times), labels


def _repeat(events, count, interval) -> pd.DataFrame:
    """Events with their additional occurrences: row i again count[i] times,
    interval[i] apart."""
    n = count.fillna(0).to_numpy().astype(int).clip(min=0)
    if not n.any():
        return events
    gap = np.zeros(len(n)) if interval is None else interval.fillna(0.0).to_numpy()
    if (gap[n > 0] <= 0).any():
        raise DataError("an event with additional occurrences needs a positive interval")
    rows = np.repeat(np.arange(len(n)), n + 1)
    rank = np.concatenate([np.arange(k + 1) for k in n])
    out = events.iloc[rows].copy()
    out["time"] = out["time"].to_numpy() + rank * gap[rows]
    for flag, plain in (("reset", 0), ("ss", 0.0)):
        if flag in out.columns:
            # the additional occurrences of a reset or steady-state record are plain doses
            out[flag] = np.where(rank > 0, plain, out[flag].to_numpy())
    return out


def _numeric(series, name) -> pd.Series:
    """Numbers, with '.' and blanks as missing; anything else is an error."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    blank = series.isna() | text.isin(_BLANK)
    out = pd.to_numeric(text.where(~blank), errors="coerce")
    bad = out.isna() & ~blank
    if bad.any():
        raise DataError(f"column {name}: non-numeric value {text[bad].iloc[0]!r}")
    return out.astype(float)


def shown(values) -> list:
    """Values as Python scalars, so that a message prints [2] and not [np.int64(2)]."""
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def _numbering(labels, names, given, kind):
    # position in names of each label, or None when the labels are already in place
    if not labels or labels == names[:len(labels)]:
        return None
    if set(labels) <= set(names):
        return np.array([names.index(lab) for lab in labels], np.int64)
    if given or len(names) <= 1 or all(_label_key(lab)[0] == 0 for lab in labels):
        return None
    raise DataError(f"the {kind} labels {labels} of the data are not the {kind}s {names} of the "
                    f"model: name them as the {kind}s, or give {kind}_labels in their order")


def _label(v) -> str:
    # 1, 1.0 and "1" name the same observation type
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(v)


def _label_key(label: str):
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def _pointers(subj: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.bincount(subj, minlength=n))]).astype(np.int64)
