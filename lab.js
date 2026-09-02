(() => {
  const formatPValue = (value) => (value < 0.0001 ? "< 0.0001" : value.toFixed(4));
  const formatEmailMean = (outcome, value) =>
    outcome === "spend" ? value.toFixed(3) : `${(value * 100).toFixed(2)}%`;
  const formatEmailEffect = (outcome, value) => {
    const sign = value >= 0 ? "+" : "";
    return outcome === "spend"
      ? `${sign}${value.toFixed(3)}`
      : `${sign}${(value * 100).toFixed(2)} pp`;
  };

  Promise.all([
    fetch("data/email-experiment.json").then((response) => {
      if (!response.ok) throw new Error("Email experiment results are unavailable.");
      return response.json();
    }),
    fetch("data/recommender-benchmark.json").then((response) => {
      if (!response.ok) throw new Error("Recommender results are unavailable.");
      return response.json();
    }),
  ])
    .then(([emailData, rankingData]) => {
      let emailArm = "Mens E-Mail";
      let emailOutcome = "spend";
      const emailLabels = {
        visit: "Visit rate",
        conversion: "Conversion rate",
        spend: "Mean spend",
      };

      const updateEmail = () => {
        const effect = emailData.primary_effects.find(
          (row) => row.treatment === emailArm && row.outcome === emailOutcome,
        );
        const treatment = emailData.arm_outcomes.find((row) => row.arm === emailArm);
        const control = emailData.arm_outcomes.find((row) => row.arm === "No E-Mail");
        const key =
          emailOutcome === "visit"
            ? "visit_rate"
            : emailOutcome === "conversion"
              ? "conversion_rate"
              : "mean_spend";
        const treatmentMean = treatment[key];
        const controlMean = control[key];
        const scaleMax = Math.max(treatmentMean, controlMean) * 1.12;

        document.querySelectorAll(".email-arm").forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.arm === emailArm));
        });
        document.querySelectorAll(".email-outcome").forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.outcome === emailOutcome));
        });

        document.querySelector("#email-lift").textContent = formatEmailEffect(
          emailOutcome,
          effect.estimate,
        );
        document.querySelector("#email-ci").textContent =
          `${formatEmailEffect(emailOutcome, effect.ci_95_low)} to ${formatEmailEffect(emailOutcome, effect.ci_95_high)}`;
        document.querySelector("#email-p").textContent = formatPValue(effect.p_value_holm);
        document.querySelector("#email-treatment-label").textContent = emailArm.replace(" E-Mail", "");
        document.querySelector("#email-treatment-value").textContent = formatEmailMean(
          emailOutcome,
          treatmentMean,
        );
        document.querySelector("#email-control-value").textContent = formatEmailMean(
          emailOutcome,
          controlMean,
        );
        document.querySelector("#email-treatment-bar").style.width =
          `${(treatmentMean / scaleMax) * 100}%`;
        document.querySelector("#email-control-bar").style.width =
          `${(controlMean / scaleMax) * 100}%`;
        document
          .querySelector("#email-bars")
          .setAttribute("aria-label", `${emailLabels[emailOutcome]} by campaign arm`);
      };

      document.querySelectorAll(".email-arm").forEach((button) => {
        button.addEventListener("click", () => {
          emailArm = button.dataset.arm;
          updateEmail();
        });
      });
      document.querySelectorAll(".email-outcome").forEach((button) => {
        button.addEventListener("click", () => {
          emailOutcome = button.dataset.outcome;
          updateEmail();
        });
      });

      let rankingMetric = "ndcg_at_10";
      let rankingSetting = "5ep_1neg";
      const rankingLabels = {
        ndcg_at_10: "NDCG@10",
        hit_rate_at_10: "HR@10",
        heldout_pairwise_accuracy: "Pairwise accuracy",
        runtime_seconds: "Training time",
        ranking_stability: "Top-10 agreement",
      };
      const getRankingValue = (setting, metric) =>
        metric === "ranking_stability"
          ? setting.ranking_stability.top10_jaccard_mean
          : setting[metric].mean;
      const formatRankingValue = (metric, value) =>
        metric === "runtime_seconds" ? `${value.toFixed(2)} s` : value.toFixed(3);

      const updateRanking = () => {
        const values = rankingData.settings.map((setting) =>
          getRankingValue(setting, rankingMetric),
        );
        const minimum =
          rankingMetric === "runtime_seconds" ? 0 : Math.min(...values) * 0.94;
        const maximum = Math.max(...values);
        const selected = rankingData.settings.find(
          (setting) => setting.setting_id === rankingSetting,
        );

        document.querySelectorAll(".ranking-metric").forEach((button) => {
          button.setAttribute(
            "aria-pressed",
            String(button.dataset.metric === rankingMetric),
          );
        });
        document.querySelectorAll(".benchmark-row").forEach((button) => {
          const setting = rankingData.settings.find(
            (item) => item.setting_id === button.dataset.setting,
          );
          const value = getRankingValue(setting, rankingMetric);
          const rawWidth =
            maximum === minimum
              ? 100
              : 18 + ((value - minimum) / (maximum - minimum)) * 82;
          const width = Math.min(100, Math.max(8, rawWidth));
          button.classList.toggle("active", setting.setting_id === rankingSetting);
          button.setAttribute(
            "aria-pressed",
            String(setting.setting_id === rankingSetting),
          );
          button.querySelector(".benchmark-track > span").style.setProperty(
            "--bar-width",
            `${width}%`,
          );
          button.querySelector("strong").textContent = formatRankingValue(
            rankingMetric,
            value,
          );
        });

        document
          .querySelector("#ranking-bars")
          .setAttribute(
            "aria-label",
            `${rankingLabels[rankingMetric]} across training settings`,
          );
        document.querySelector("#ranking-setting-label").textContent =
          `${selected.epochs} epochs · ${selected.negatives_per_positive} negative${selected.negatives_per_positive > 1 ? "s" : ""} per positive`;
        document.querySelector("#ranking-updates").textContent =
          `${selected.updates.toLocaleString()} updates`;

        let range = "descriptive mean across seed pairs";
        if (rankingMetric !== "ranking_stability") {
          range =
            `${formatRankingValue(rankingMetric, selected[rankingMetric].min)}–${formatRankingValue(rankingMetric, selected[rankingMetric].max)} across seeds`;
        }
        document.querySelector("#ranking-range").textContent = range;
      };

      document.querySelectorAll(".ranking-metric").forEach((button) => {
        button.addEventListener("click", () => {
          rankingMetric = button.dataset.metric;
          updateRanking();
        });
      });
      document.querySelectorAll(".benchmark-row").forEach((button) => {
        button.addEventListener("click", () => {
          rankingSetting = button.dataset.setting;
          updateRanking();
        });
      });

      updateEmail();
      updateRanking();
    })
    .catch(() => {
      // The server-rendered initial results remain visible if JSON loading fails.
    });
})();
