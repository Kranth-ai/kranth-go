package kranth

import (
	"encoding/json"
	"testing"
)

// TestParseDebateDetail verifies that the flattened GET /v1/debates/:id payload
// is correctly split into DebateRow, Synthesis, and Audience without manual
// field extraction.
func TestParseDebateDetail(t *testing.T) {
	score := 72
	payload := map[string]any{
		// DebateRow fields (promoted by embedded struct)
		"id":            "dbt_abc123",
		"topic":         "Should B2B SaaS default to annual billing?",
		"mode":          "parliament",
		"status":        "complete",
		"participants":  6,
		"max_turns":     12,
		"duration_secs": 340,
		"model_id":      "claude-sonnet-4-6",
		"error_message": nil,
		"started_at":    "2026-06-20T10:00:00Z",
		"completed_at":  "2026-06-20T10:05:40Z",
		"created_at":    "2026-06-20T09:59:55Z",
		"public":        true,
		"turn_count":    12,
		"voice_enabled": false,
		"score":         score,
		// Extra fields at same level (not in DebateRow)
		"synthesis": map[string]any{
			"score":             72,
			"pct_for":           58,
			"pct_against":       29,
			"pct_undecided":     13,
			"verdict_reasons":   []string{"Strong cash-flow argument", "Annual locks reduce churn"},
			"decisive_argument": "Predictable ARR outweighs new-customer friction",
			"top_unrebutted":    "Monthly gives buyers budget flexibility",
			"summary":           "The panel narrowly endorsed annual-first pricing",
			"key_arguments":     []string{"ARR smooths forecasting", "Upfront cash funds growth"},
			"key_objections":    []string{"Monthly lowers signup friction"},
			"common_ground":     []string{"Billing cadence matters less than perceived value"},
			"strongest_turn":    "The CFO persona on cash-flow predictability",
			"model_id":          "claude-sonnet-4-6",
			"created_at":        "2026-06-20T10:05:38Z",
		},
		"audience": []map[string]any{
			{
				"persona_id":      "p_001",
				"speaker_name":    "Grace",
				"archetype":       "pragmatic-operator",
				"body":            "Annual makes sense once PMF is proven.",
				"sentiment":       0.65,
				"sentiment_label": "positive",
				"turn_seq":        4,
			},
		},
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}

	detail, err := parseDebateDetail(raw)
	if err != nil {
		t.Fatalf("parseDebateDetail: %v", err)
	}

	// DebateRow fields promoted correctly.
	if detail.Debate.ID != "dbt_abc123" {
		t.Errorf("debate ID: got %q, want %q", detail.Debate.ID, "dbt_abc123")
	}
	if detail.Debate.Score == nil || *detail.Debate.Score != 72 {
		t.Errorf("debate score: got %v, want 72", detail.Debate.Score)
	}
	if detail.Debate.Status != "complete" {
		t.Errorf("debate status: got %q, want %q", detail.Debate.Status, "complete")
	}

	// Synthesis parsed.
	if detail.Synthesis == nil {
		t.Fatal("synthesis is nil")
	}
	if detail.Synthesis.Score != 72 {
		t.Errorf("synthesis score: got %d, want 72", detail.Synthesis.Score)
	}
	if detail.Synthesis.PctFor != 58 {
		t.Errorf("synthesis pct_for: got %d, want 58", detail.Synthesis.PctFor)
	}
	if len(detail.Synthesis.VerdictReasons) != 2 {
		t.Errorf("verdict_reasons len: got %d, want 2", len(detail.Synthesis.VerdictReasons))
	}

	// Audience parsed.
	if len(detail.Audience) != 1 {
		t.Fatalf("audience len: got %d, want 1", len(detail.Audience))
	}
	a := detail.Audience[0]
	if a.PersonaID != "p_001" {
		t.Errorf("audience persona_id: got %q", a.PersonaID)
	}
	if a.Sentiment != 0.65 {
		t.Errorf("audience sentiment: got %f, want 0.65", a.Sentiment)
	}
	if a.TurnSeq != 4 {
		t.Errorf("audience turn_seq: got %d, want 4", a.TurnSeq)
	}

	// Nil synthesis case.
	payloadNoSynth := map[string]any{
		"id": "dbt_xyz", "topic": "t", "mode": "m", "status": "queued",
		"participants": 2, "max_turns": 4, "duration_secs": 0,
		"model_id": "haiku", "created_at": "2026-06-20T00:00:00Z",
		"public": false, "turn_count": 0, "voice_enabled": false,
	}
	rawNoSynth, _ := json.Marshal(payloadNoSynth)
	detailNoSynth, err := parseDebateDetail(rawNoSynth)
	if err != nil {
		t.Fatalf("parseDebateDetail (no synth): %v", err)
	}
	if detailNoSynth.Synthesis != nil {
		t.Error("synthesis should be nil when absent from payload")
	}
	if len(detailNoSynth.Audience) != 0 {
		t.Errorf("audience should be empty slice, got len %d", len(detailNoSynth.Audience))
	}
}
