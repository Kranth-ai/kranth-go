// Package kranth is the official Go client for the Kranth API.
//
// Stress-test ideas against hundreds of AI personas. Adversarial multi-bias
// debate. API-first.
//
//	client := kranth.New("kr_live_…")
//
//	// Simulation
//	sim, err := client.Sims.Create(ctx, kranth.CreateSimRequest{
//	    IdeaText:     "Launch a paid Rust devtool. $29/mo.",
//	    PersonaCount: 50,
//	    ModelID:      "claude-sonnet-4-6",
//	})
//
//	for ev := range client.Sims.Stream(ctx, sim.SimID) {
//	    if ev.Err != nil { return ev.Err }
//	    switch ev.Event.Kind {
//	    case "persona.ready":  // ...
//	    case "sim.complete":   // ...
//	    }
//	}
//
//	// Recon (web-grounded research)
//	run, err := client.Recon.Create(ctx, kranth.CreateReconRequest{
//	    IdeaText: "Vertical SaaS for veterinary clinics",
//	    Tier:     "standard",
//	})
//
//	for ev := range client.Recon.Stream(ctx, run.ReconID) {
//	    if ev.Err != nil { return ev.Err }
//	    switch ev.Event.Kind {
//	    case "recon.complete": // ...
//	    }
//	}
//
//	// Debates (adversarial panel)
//	debate, err := client.Debates.Create(ctx, kranth.CreateDebateRequest{
//	    Topic: "Should B2B SaaS default to annual billing?",
//	    Mode:  "parliament",
//	})
//
//	for ev := range client.Debates.Stream(ctx, debate.ID) {
//	    if ev.Err != nil { return ev.Err }
//	    switch ev.Event.Kind {
//	    case "debate.complete": // ...
//	    }
//	}
package kranth

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Version of the SDK — kept in sync with the git tag at release time.
const Version = "0.3.0"

const (
	DefaultBaseURL = "https://api.kranth.ai"
	userAgent      = "kranth-go/" + Version
)

// Client is the entry point.  One client per process is fine; the underlying
// net/http transport multiplexes goroutine-safely.
type Client struct {
	APIKey  string
	BaseURL string
	HTTP    *http.Client

	Sims    *SimsService
	Recon   *ReconService
	Debates *DebatesService
	APIKeys *APIKeysService
	Models  *ModelsService
	Billing *BillingService
}

// New constructs a client with sensible defaults.  Pass options to override.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		APIKey:  apiKey,
		BaseURL: DefaultBaseURL,
		HTTP:    &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(c)
	}
	c.BaseURL = strings.TrimRight(c.BaseURL, "/")
	c.Sims = &SimsService{c: c}
	c.Recon = &ReconService{c: c}
	c.Debates = &DebatesService{c: c}
	c.APIKeys = &APIKeysService{c: c}
	c.Models = &ModelsService{c: c}
	c.Billing = &BillingService{c: c}
	return c
}

// Option mutates a Client at construction. See WithBaseURL / WithHTTPClient.
type Option func(*Client)

func WithBaseURL(u string) Option {
	return func(c *Client) { c.BaseURL = u }
}

func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) { c.HTTP = h }
}

// ─── error type ─────────────────────────────────────────────

// APIError is what every non-2xx response decodes to. RetryAfter is set when
// the server returned a Retry-After header (only on 429).
type APIError struct {
	StatusCode int
	Code       string
	Message    string
	Body       json.RawMessage
	RetryAfter *float64
}

func (e *APIError) Error() string {
	return fmt.Sprintf("kranth: [%d] %s: %s", e.StatusCode, e.Code, e.Message)
}

// IsAuthError reports whether err is a 401.
func IsAuthError(err error) bool {
	var a *APIError
	return errors.As(err, &a) && a.StatusCode == 401
}

// IsPaymentRequired reports whether err is a 402 (out of credits).
func IsPaymentRequired(err error) bool {
	var a *APIError
	return errors.As(err, &a) && a.StatusCode == 402
}

// IsRateLimited reports whether err is a 429 (over RPM).
func IsRateLimited(err error) bool {
	var a *APIError
	return errors.As(err, &a) && a.StatusCode == 429
}

// ─── types ──────────────────────────────────────────────────

type SimStatus string

const (
	SimStatusQueued   SimStatus = "queued"
	SimStatusRunning  SimStatus = "running"
	SimStatusComplete SimStatus = "complete"
	SimStatusFailed   SimStatus = "failed"
	SimStatusCanceled SimStatus = "canceled"
)

type VerdictReport struct {
	SentimentScore       float64  `json:"sentiment_score"`
	HostileCount         uint32   `json:"hostile_count"`
	AlignedCount         uint32   `json:"aligned_count"`
	NeutralCount         uint32   `json:"neutral_count"`
	Themes               []string `json:"themes"`
	TopObjections        []string `json:"top_objections"`
	RepresentativeQuotes []string `json:"representative_quotes"`
}

type SimSummary struct {
	ID             string         `json:"id"`
	OrgID          string         `json:"org_id"`
	CreatedBy      string         `json:"created_by"`
	Status         SimStatus      `json:"status"`
	IdeaText       string         `json:"idea_text"`
	PersonaCount   int            `json:"persona_count"`
	PersonasDone   int            `json:"personas_done"`
	ModelTier      string         `json:"model_tier"`
	CreditsCharged int            `json:"credits_charged"`
	AvgSentiment   *float64       `json:"avg_sentiment"`
	ErrorMessage   *string        `json:"error_message"`
	StartedAt      *string        `json:"started_at"`
	CompletedAt    *string        `json:"completed_at"`
	CreatedAt      string         `json:"created_at"`
	Report         *VerdictReport `json:"report,omitempty"`
}

type CreateSimRequest struct {
	IdeaText       string   `json:"idea_text"`
	PersonaCount   uint32   `json:"persona_count"`
	ModelID        string   `json:"model_id"`
	TrainingOptIn  *bool    `json:"training_opt_in,omitempty"`
	ConfidenceMode *bool    `json:"confidence_mode,omitempty"`
	BiasAxes       []string `json:"bias_axes,omitempty"`
	IdempotencyKey string   `json:"-"`
}

type CreateSimResponse struct {
	SimID          string `json:"sim_id"`
	Status         string `json:"status"`
	ModelTier      string `json:"model_tier"`
	CreditsCharged uint32 `json:"credits_charged"`
	PersonaCount   uint32 `json:"persona_count"`
}

type ListSimsResponse struct {
	Sims       []SimSummary `json:"sims"`
	NextCursor *string      `json:"next_cursor"`
}

type ListSimsParams struct {
	Status string
	Cursor string
	Limit  int
}

type APIKey struct {
	ID         string  `json:"id"`
	Name       string  `json:"name"`
	Prefix     string  `json:"prefix"`
	LastUsedAt *string `json:"last_used_at"`
	RevokedAt  *string `json:"revoked_at"`
	CreatedAt  string  `json:"created_at"`
}

type CreateAPIKeyResponse struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Prefix    string `json:"prefix"`
	Key       string `json:"key"`
	CreatedAt string `json:"created_at"`
}

type ModelInfo struct {
	ID                    string `json:"id"`
	Tier                  string `json:"tier"`
	DisplayName           string `json:"display_name"`
	Provider              string `json:"provider"`
	CreditsPer100Personas uint32 `json:"credits_per_100_personas"`
	PlanMin               string `json:"plan_min"`
}

type Usage struct {
	Plan             string `json:"plan"`
	MonthlyCap       int    `json:"monthly_cap"`
	CreditsConsumed  int    `json:"credits_consumed"`
	CreditsRemaining int64  `json:"credits_remaining"`
	PeriodStart      string `json:"period_start"`
}

type Me struct {
	UserID     string `json:"user_id"`
	OrgID      string `json:"org_id"`
	AuthSource string `json:"auth_source"`
}

// SimEvent is one frame off the SSE stream. Data is opaque JSON; the field
// shape depends on Kind.
type SimEvent struct {
	Kind  string          `json:"kind"`
	SimID string          `json:"sim_id"`
	Data  json.RawMessage `json:"data,omitempty"`
}

// StreamFrame wraps SimEvent with the stream-side error so a single channel
// communicates both success and failure.
type StreamFrame struct {
	Event SimEvent
	Err   error
}

// ─── shared request helper ──────────────────────────────────

func (c *Client) request(ctx context.Context, method, path string, body, out any, idempKey string) error {
	var reader io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}
		reader = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", userAgent)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if idempKey != "" {
		req.Header.Set("Idempotency-Key", idempKey)
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		apiErr := &APIError{
			StatusCode: resp.StatusCode,
			Code:       asString(parsed["error"], fmt.Sprintf("http_%d", resp.StatusCode)),
			Message:    asString(parsed["message"], resp.Status),
			Body:       raw,
		}
		if resp.StatusCode == 429 {
			if h := resp.Header.Get("Retry-After"); h != "" {
				if f, err := strconv.ParseFloat(h, 64); err == nil {
					apiErr.RetryAfter = &f
				}
			}
		}
		return apiErr
	}

	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	}
	return nil
}

func asString(v any, fallback string) string {
	if s, ok := v.(string); ok {
		return s
	}
	return fallback
}

// ─── resources ──────────────────────────────────────────────

type SimsService struct{ c *Client }

func (s *SimsService) Create(ctx context.Context, req CreateSimRequest) (*CreateSimResponse, error) {
	out := &CreateSimResponse{}
	if err := s.c.request(ctx, "POST", "/v1/sims", req, out, req.IdempotencyKey); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SimsService) Get(ctx context.Context, simID string) (*SimSummary, error) {
	out := &SimSummary{}
	if err := s.c.request(ctx, "GET", "/v1/sims/"+url.PathEscape(simID), nil, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SimsService) List(ctx context.Context, params ListSimsParams) (*ListSimsResponse, error) {
	q := url.Values{}
	if params.Status != "" {
		q.Set("status", params.Status)
	}
	if params.Cursor != "" {
		q.Set("cursor", params.Cursor)
	}
	if params.Limit > 0 {
		q.Set("limit", strconv.Itoa(params.Limit))
	}
	path := "/v1/sims"
	if encoded := q.Encode(); encoded != "" {
		path += "?" + encoded
	}
	out := &ListSimsResponse{}
	if err := s.c.request(ctx, "GET", path, nil, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SimsService) Cancel(ctx context.Context, simID string) error {
	return s.c.request(ctx, "POST", "/v1/sims/"+url.PathEscape(simID)+"/cancel", nil, nil, "")
}

func (s *SimsService) Export(ctx context.Context, simID string) (json.RawMessage, error) {
	var out json.RawMessage
	if err := s.c.request(ctx, "GET", "/v1/sims/"+url.PathEscape(simID)+"/export", nil, &out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

// Stream subscribes to the sim's SSE event stream. Send is closed when the
// server hangs up (typically after sim.complete or sim.failed) or when ctx
// is canceled. A non-nil StreamFrame.Err is the last item written.
func (s *SimsService) Stream(ctx context.Context, simID string) <-chan StreamFrame {
	out := make(chan StreamFrame, 8)
	go func() {
		defer close(out)
		// Mint a short-lived stream token (EventSource can't send Bearer headers).
		var tokenResp struct {
			Token string `json:"token"`
		}
		if err := s.c.request(ctx, "POST", "/v1/sims/"+url.PathEscape(simID)+"/stream-token", nil, &tokenResp, ""); err != nil {
			out <- StreamFrame{Err: err}
			return
		}
		eventsURL := s.c.BaseURL + "/v1/sims/" + url.PathEscape(simID) + "/events?token=" + url.QueryEscape(tokenResp.Token)
		req, err := http.NewRequestWithContext(ctx, "GET", eventsURL, nil)
		if err != nil {
			out <- StreamFrame{Err: err}
			return
		}
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("User-Agent", userAgent)
		resp, err := s.c.HTTP.Do(req)
		if err != nil {
			out <- StreamFrame{Err: err}
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			raw, _ := io.ReadAll(resp.Body)
			out <- StreamFrame{Err: &APIError{StatusCode: resp.StatusCode, Code: "stream_open_failed", Message: string(raw), Body: raw}}
			return
		}

		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var dataLines []string
		flush := func() {
			if len(dataLines) == 0 {
				return
			}
			payload := strings.Join(dataLines, "\n")
			dataLines = nil
			if payload == "" || payload == "ping" {
				return
			}
			var ev SimEvent
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				return // skip non-JSON keep-alives
			}
			select {
			case out <- StreamFrame{Event: ev}:
			case <-ctx.Done():
			}
		}
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				flush()
				continue
			}
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
		flush()
		if err := sc.Err(); err != nil {
			out <- StreamFrame{Err: err}
		}
	}()
	return out
}

type APIKeysService struct{ c *Client }

func (s *APIKeysService) Create(ctx context.Context, name, env string) (*CreateAPIKeyResponse, error) {
	if env == "" {
		env = "live"
	}
	out := &CreateAPIKeyResponse{}
	if err := s.c.request(ctx, "POST", "/v1/api-keys", map[string]string{"name": name, "env": env}, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *APIKeysService) List(ctx context.Context) ([]APIKey, error) {
	out := struct {
		Keys []APIKey `json:"keys"`
	}{}
	if err := s.c.request(ctx, "GET", "/v1/api-keys", nil, &out, ""); err != nil {
		return nil, err
	}
	return out.Keys, nil
}

func (s *APIKeysService) Revoke(ctx context.Context, id string) error {
	return s.c.request(ctx, "DELETE", "/v1/api-keys/"+url.PathEscape(id), nil, nil, "")
}

type ModelsService struct{ c *Client }

func (s *ModelsService) List(ctx context.Context) ([]ModelInfo, error) {
	out := struct {
		Models []ModelInfo `json:"models"`
	}{}
	if err := s.c.request(ctx, "GET", "/v1/models", nil, &out, ""); err != nil {
		return nil, err
	}
	return out.Models, nil
}

type BillingService struct{ c *Client }

func (s *BillingService) Usage(ctx context.Context) (*Usage, error) {
	out := &Usage{}
	if err := s.c.request(ctx, "GET", "/v1/usage", nil, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

type CheckoutParams struct {
	Plan          string // "starter" | "pro" | "scale"
	BillingPeriod string // "monthly" | "annual"; default monthly
	SuccessURL    string
	CancelURL     string
}

func (s *BillingService) CheckoutURL(ctx context.Context, p CheckoutParams) (string, error) {
	body := map[string]any{"plan": p.Plan}
	if p.BillingPeriod != "" {
		body["billing_period"] = p.BillingPeriod
	}
	if p.SuccessURL != "" {
		body["success_url"] = p.SuccessURL
	}
	if p.CancelURL != "" {
		body["cancel_url"] = p.CancelURL
	}
	var out struct {
		URL string `json:"url"`
	}
	if err := s.c.request(ctx, "POST", "/v1/billing/checkout", body, &out, ""); err != nil {
		return "", err
	}
	return out.URL, nil
}

func (s *BillingService) PortalURL(ctx context.Context, returnURL string) (string, error) {
	body := map[string]any{}
	if returnURL != "" {
		body["return_url"] = returnURL
	}
	var out struct {
		URL string `json:"url"`
	}
	if err := s.c.request(ctx, "POST", "/v1/billing/portal", body, &out, ""); err != nil {
		return "", err
	}
	return out.URL, nil
}

// Me returns the authenticated principal.
func (c *Client) Me(ctx context.Context) (*Me, error) {
	out := &Me{}
	if err := c.request(ctx, "GET", "/v1/me", nil, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

// ─── Recon types ─────────────────────────────────────────────────

// CreateReconRequest is the body for POST /v1/recon.
// IdeaText and Tier are required; all others are optional.
type CreateReconRequest struct {
	IdeaText       string  `json:"idea_text"`
	Tier           string  `json:"tier"`
	WorkerModel    string  `json:"worker_model,omitempty"`
	SynthModel     string  `json:"synth_model,omitempty"`
	Audience       string  `json:"audience,omitempty"`
	CompanyURL     string  `json:"company_url,omitempty"`
	IdempotencyKey string  `json:"-"`
}

// CreateReconResponse is returned by POST /v1/recon.
type CreateReconResponse struct {
	ReconID        string `json:"recon_id"`
	Status         string `json:"status"`
	Tier           string `json:"tier"`
	CreditsCharged int    `json:"credits_charged"`
}

// ReconSummary is one entry in the GET /v1/recon list.
type ReconSummary struct {
	ID             string   `json:"id"`
	Status         string   `json:"status"`
	IdeaText       string   `json:"idea_text"`
	Tier           string   `json:"tier"`
	Score          *int     `json:"score"`
	CreditsCharged int      `json:"credits_charged"`
	CreatedAt      string   `json:"created_at"`
	CompletedAt    *string  `json:"completed_at"`
	Supported      int      `json:"supported"`
	Contradicted   int      `json:"contradicted"`
	Neutral        int      `json:"neutral"`
	SourceDomains  []string `json:"source_domains"`
}

// ReconTier describes one tier from GET /v1/recon/tiers.
type ReconTier struct {
	Slug         string `json:"slug"`
	Label        string `json:"label"`
	WorkerCount  int    `json:"worker_count"`
	PriceCredits int    `json:"price_credits"`
}

// ReconRun is the run object inside GET /v1/recon/:id.
type ReconRun struct {
	ID             string             `json:"id"`
	Status         string             `json:"status"`
	IdeaText       string             `json:"idea_text"`
	Tier           string             `json:"tier"`
	Score          *int               `json:"score"`
	CreditsCharged int                `json:"credits_charged"`
	CreatedAt      string             `json:"created_at"`
	CompletedAt    *string            `json:"completed_at"`
	Report         json.RawMessage    `json:"report"`
	ModelConfig    map[string]any     `json:"model_config"`
}

// ReconFinding is one agent finding within a recon run.
type ReconFinding struct {
	ID        string `json:"id"`
	AgentRole string `json:"agent_role"`
	Claim     string `json:"claim"`
	Grounding int    `json:"grounding"` // 1 supported · 0 neutral · -1 contradicted
}

// ReconSource is a cited web source attached to a finding.
type ReconSource struct {
	FindingID *string `json:"finding_id"`
	URL       string  `json:"url"`
	Title     string  `json:"title"`
	Snippet   string  `json:"snippet"`
}

// ReconDetail is the full response from GET /v1/recon/:id.
type ReconDetail struct {
	Run      ReconRun       `json:"run"`
	Findings []ReconFinding `json:"findings"`
	Sources  []ReconSource  `json:"sources"`
}

// ReconEvent is one frame off the recon SSE stream.
type ReconEvent struct {
	Kind    string          `json:"kind"`
	ReconID string          `json:"recon_id"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// ReconStreamFrame wraps ReconEvent with the stream-side error.
type ReconStreamFrame struct {
	Event ReconEvent
	Err   error
}

// ─── Recon service ────────────────────────────────────────────────

type ReconService struct{ c *Client }

func (s *ReconService) Create(ctx context.Context, req CreateReconRequest) (*CreateReconResponse, error) {
	out := &CreateReconResponse{}
	if err := s.c.request(ctx, "POST", "/v1/recon", req, out, req.IdempotencyKey); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *ReconService) Get(ctx context.Context, reconID string) (*ReconDetail, error) {
	out := &ReconDetail{}
	if err := s.c.request(ctx, "GET", "/v1/recon/"+url.PathEscape(reconID), nil, out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *ReconService) List(ctx context.Context) ([]ReconSummary, error) {
	out := struct {
		Recons []ReconSummary `json:"recons"`
	}{}
	if err := s.c.request(ctx, "GET", "/v1/recon", nil, &out, ""); err != nil {
		return nil, err
	}
	return out.Recons, nil
}

func (s *ReconService) Tiers(ctx context.Context) ([]ReconTier, error) {
	out := struct {
		Tiers []ReconTier `json:"tiers"`
	}{}
	if err := s.c.request(ctx, "GET", "/v1/recon/tiers", nil, &out, ""); err != nil {
		return nil, err
	}
	return out.Tiers, nil
}

func (s *ReconService) Export(ctx context.Context, reconID string) (json.RawMessage, error) {
	var out json.RawMessage
	if err := s.c.request(ctx, "GET", "/v1/recon/"+url.PathEscape(reconID)+"/export", nil, &out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

// Stream subscribes to the recon run's SSE event stream. The channel is closed
// when the server hangs up (after recon.complete or recon.failed) or when ctx
// is canceled. A non-nil ReconStreamFrame.Err is the last item written.
func (s *ReconService) Stream(ctx context.Context, reconID string) <-chan ReconStreamFrame {
	out := make(chan ReconStreamFrame, 8)
	go func() {
		defer close(out)
		var tokenResp struct {
			Token string `json:"token"`
		}
		if err := s.c.request(ctx, "POST", "/v1/recon/"+url.PathEscape(reconID)+"/stream-token", nil, &tokenResp, ""); err != nil {
			out <- ReconStreamFrame{Err: err}
			return
		}
		eventsURL := s.c.BaseURL + "/v1/recon/" + url.PathEscape(reconID) + "/events?token=" + url.QueryEscape(tokenResp.Token)
		req, err := http.NewRequestWithContext(ctx, "GET", eventsURL, nil)
		if err != nil {
			out <- ReconStreamFrame{Err: err}
			return
		}
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("User-Agent", userAgent)
		resp, err := s.c.HTTP.Do(req)
		if err != nil {
			out <- ReconStreamFrame{Err: err}
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			raw, _ := io.ReadAll(resp.Body)
			out <- ReconStreamFrame{Err: &APIError{StatusCode: resp.StatusCode, Code: "stream_open_failed", Message: string(raw), Body: raw}}
			return
		}

		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var dataLines []string
		flush := func() {
			if len(dataLines) == 0 {
				return
			}
			payload := strings.Join(dataLines, "\n")
			dataLines = nil
			if payload == "" || payload == "ping" {
				return
			}
			var ev ReconEvent
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				return
			}
			select {
			case out <- ReconStreamFrame{Event: ev}:
			case <-ctx.Done():
			}
		}
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				flush()
				continue
			}
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
		flush()
		if err := sc.Err(); err != nil {
			out <- ReconStreamFrame{Err: err}
		}
	}()
	return out
}

// ─── Debates types ───────────────────────────────────────────────

// CreateDebateRequest is the body for POST /v1/debates.
// Topic and Mode are required; all others are optional.
type CreateDebateRequest struct {
	Topic          string   `json:"topic"`
	Mode           string   `json:"mode"`
	Participants   *int     `json:"participants,omitempty"`
	ModelID        string   `json:"model_id,omitempty"`
	TrainingOptIn  *bool    `json:"training_opt_in,omitempty"`
	Public         *bool    `json:"public,omitempty"`
	VoiceEnabled   *bool    `json:"voice_enabled,omitempty"`
	ArchetypeSlugs []string `json:"archetype_slugs,omitempty"`
	IdempotencyKey string   `json:"-"`
}

// DebateRow is the canonical debate shape returned by Create, List, and
// embedded in DebateDetail.
type DebateRow struct {
	ID            string  `json:"id"`
	Topic         string  `json:"topic"`
	Mode          string  `json:"mode"`
	Status        string  `json:"status"`
	Participants  int     `json:"participants"`
	MaxTurns      int     `json:"max_turns"`
	DurationSecs  int     `json:"duration_secs"`
	ModelID       string  `json:"model_id"`
	ErrorMessage  *string `json:"error_message"`
	StartedAt     *string `json:"started_at"`
	CompletedAt   *string `json:"completed_at"`
	CreatedAt     string  `json:"created_at"`
	Public        bool    `json:"public"`
	TurnCount     int     `json:"turn_count"`
	VoiceEnabled  bool    `json:"voice_enabled"`
	Score         *int    `json:"score"`
}

// SpeakerPreview is the card avatar preview attached to each debate in list.
type SpeakerPreview struct {
	SpeakerName      string `json:"speaker_name"`
	SpeakerUsername  string `json:"speaker_username"`
	SpeakerArchetype string `json:"speaker_archetype"`
}

// ListDebatesResponse is returned by GET /v1/debates.
type ListDebatesResponse struct {
	Debates  []DebateRow                 `json:"debates"`
	Speakers map[string][]SpeakerPreview `json:"speakers"`
}

// DebateTurn is one speaker turn from GET /v1/debates/:id/turns.
type DebateTurn struct {
	SpeakerID        string  `json:"speaker_id"`
	SpeakerArchetype string  `json:"speaker_archetype"`
	SpeakerName      string  `json:"speaker_name"`
	SpeakerUsername  string  `json:"speaker_username"`
	Role             string  `json:"role"`
	Body             string  `json:"body"`
	EmittedAt        string  `json:"emitted_at"`
	AudioURL         *string `json:"audio_url"`
	AudioVoiceID     *string `json:"audio_voice_id"`
	AudioMs          *int    `json:"audio_ms"`
}

// DebateSynthesis is the verdict spine from the detail endpoint.
type DebateSynthesis struct {
	Score             int      `json:"score"`
	PctFor            int      `json:"pct_for"`
	PctAgainst        int      `json:"pct_against"`
	PctUndecided      int      `json:"pct_undecided"`
	VerdictReasons    []string `json:"verdict_reasons"`
	DecisiveArgument  string   `json:"decisive_argument"`
	TopUnrebutted     string   `json:"top_unrebutted"`
	Summary           string   `json:"summary"`
	KeyArguments      []string `json:"key_arguments"`
	KeyObjections     []string `json:"key_objections"`
	CommonGround      []string `json:"common_ground"`
	StrongestTurn     string   `json:"strongest_turn"`
	ModelID           string   `json:"model_id"`
	CreatedAt         string   `json:"created_at"`
}

// AudienceReaction is one persona reaction from the detail endpoint.
type AudienceReaction struct {
	PersonaID      string  `json:"persona_id"`
	SpeakerName    string  `json:"speaker_name"`
	Archetype      string  `json:"archetype"`
	Body           string  `json:"body"`
	Sentiment      float64 `json:"sentiment"`
	SentimentLabel string  `json:"sentiment_label"`
	TurnSeq        int     `json:"turn_seq"`
}

// DebateDetail is the structured response from GET /v1/debates/:id (and PATCH).
// The API returns a flattened payload (debate fields + synthesis + audience at
// the top level); we parse it cleanly using an embedded-struct wire type.
type DebateDetail struct {
	Debate    DebateRow        `json:"-"`
	Synthesis *DebateSynthesis `json:"-"`
	Audience  []AudienceReaction `json:"-"`
}

// debateDetailWire is the embedded-struct trick for the flattened payload.
// Go's encoding/json handles embedded structs by promoting all fields into the
// parent, so DebateRow fields decode directly without manual field extraction.
type debateDetailWire struct {
	DebateRow
	Synthesis *DebateSynthesis   `json:"synthesis"`
	Audience  []AudienceReaction `json:"audience"`
}

func parseDebateDetail(raw []byte) (*DebateDetail, error) {
	var wire debateDetailWire
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("decode debate detail: %w", err)
	}
	audience := wire.Audience
	if audience == nil {
		audience = []AudienceReaction{}
	}
	return &DebateDetail{
		Debate:    wire.DebateRow,
		Synthesis: wire.Synthesis,
		Audience:  audience,
	}, nil
}

// DebateEvent is one frame off the debate SSE stream.
type DebateEvent struct {
	Kind     string          `json:"kind"`
	DebateID string          `json:"debate_id"`
	Data     json.RawMessage `json:"data,omitempty"`
}

// DebateStreamFrame wraps DebateEvent with the stream-side error.
type DebateStreamFrame struct {
	Event DebateEvent
	Err   error
}

// ─── Debates service ─────────────────────────────────────────────

type DebatesService struct{ c *Client }

func (s *DebatesService) Create(ctx context.Context, req CreateDebateRequest) (*DebateRow, error) {
	out := &DebateRow{}
	if err := s.c.request(ctx, "POST", "/v1/debates", req, out, req.IdempotencyKey); err != nil {
		return nil, err
	}
	return out, nil
}

// Get returns the full debate detail (flattened debate row + synthesis + audience).
func (s *DebatesService) Get(ctx context.Context, debateID string) (*DebateDetail, error) {
	return s.getDetail(ctx, "GET", "/v1/debates/"+url.PathEscape(debateID), nil)
}

func (s *DebatesService) List(ctx context.Context) (*ListDebatesResponse, error) {
	out := &ListDebatesResponse{}
	if err := s.c.request(ctx, "GET", "/v1/debates", nil, out, ""); err != nil {
		return nil, err
	}
	if out.Speakers == nil {
		out.Speakers = map[string][]SpeakerPreview{}
	}
	return out, nil
}

func (s *DebatesService) Turns(ctx context.Context, debateID string) ([]DebateTurn, error) {
	out := struct {
		Turns []DebateTurn `json:"turns"`
	}{}
	if err := s.c.request(ctx, "GET", "/v1/debates/"+url.PathEscape(debateID)+"/turns", nil, &out, ""); err != nil {
		return nil, err
	}
	return out.Turns, nil
}

// SetPublic patches the debate's public flag and returns the updated detail.
func (s *DebatesService) SetPublic(ctx context.Context, debateID string, public bool) (*DebateDetail, error) {
	return s.getDetail(ctx, "PATCH", "/v1/debates/"+url.PathEscape(debateID), map[string]bool{"public": public})
}

func (s *DebatesService) Cancel(ctx context.Context, debateID string) error {
	return s.c.request(ctx, "POST", "/v1/debates/"+url.PathEscape(debateID)+"/cancel", nil, nil, "")
}

func (s *DebatesService) Export(ctx context.Context, debateID string) (json.RawMessage, error) {
	var out json.RawMessage
	if err := s.c.request(ctx, "GET", "/v1/debates/"+url.PathEscape(debateID)+"/export", nil, &out, ""); err != nil {
		return nil, err
	}
	return out, nil
}

// getDetail is the shared helper for Get and SetPublic — both return the
// flattened debate-detail payload that needs parseDebateDetail.
func (s *DebatesService) getDetail(ctx context.Context, method, path string, body any) (*DebateDetail, error) {
	var raw json.RawMessage
	if err := s.c.request(ctx, method, path, body, &raw, ""); err != nil {
		return nil, err
	}
	return parseDebateDetail(raw)
}

// Stream subscribes to the debate's SSE event stream. The channel is closed
// when the server hangs up (after debate.complete or debate.failed) or when ctx
// is canceled. A non-nil DebateStreamFrame.Err is the last item written.
func (s *DebatesService) Stream(ctx context.Context, debateID string) <-chan DebateStreamFrame {
	out := make(chan DebateStreamFrame, 8)
	go func() {
		defer close(out)
		var tokenResp struct {
			Token string `json:"token"`
		}
		if err := s.c.request(ctx, "POST", "/v1/debates/"+url.PathEscape(debateID)+"/stream-token", nil, &tokenResp, ""); err != nil {
			out <- DebateStreamFrame{Err: err}
			return
		}
		eventsURL := s.c.BaseURL + "/v1/debates/" + url.PathEscape(debateID) + "/events?token=" + url.QueryEscape(tokenResp.Token)
		req, err := http.NewRequestWithContext(ctx, "GET", eventsURL, nil)
		if err != nil {
			out <- DebateStreamFrame{Err: err}
			return
		}
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("User-Agent", userAgent)
		resp, err := s.c.HTTP.Do(req)
		if err != nil {
			out <- DebateStreamFrame{Err: err}
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			raw, _ := io.ReadAll(resp.Body)
			out <- DebateStreamFrame{Err: &APIError{StatusCode: resp.StatusCode, Code: "stream_open_failed", Message: string(raw), Body: raw}}
			return
		}

		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var dataLines []string
		flush := func() {
			if len(dataLines) == 0 {
				return
			}
			payload := strings.Join(dataLines, "\n")
			dataLines = nil
			if payload == "" || payload == "ping" {
				return
			}
			var ev DebateEvent
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				return
			}
			select {
			case out <- DebateStreamFrame{Event: ev}:
			case <-ctx.Done():
			}
		}
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				flush()
				continue
			}
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
		flush()
		if err := sc.Err(); err != nil {
			out <- DebateStreamFrame{Err: err}
		}
	}()
	return out
}
