// Package kranth is the official Go client for the Kranth API.
//
// Stress-test ideas against hundreds of AI personas. Adversarial multi-bias
// debate. API-first.
//
//	client := kranth.New("kr_live_…")
//
//	sim, err := client.Sims.Create(ctx, kranth.CreateSimRequest{
//	    IdeaText:     "Launch a paid Rust devtool. $29/mo.",
//	    PersonaCount: 50,
//	    ModelID:      "claude-sonnet-4-6",
//	})
//
//	for ev := range client.Sims.Stream(ctx, sim.SimID) {
//	    if ev.Err != nil { return ev.Err }
//	    switch ev.Event.Kind {
//	    case "persona.ready":     // ...
//	    case "sim.complete":      // ...
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
const Version = "0.1.0"

const (
	DefaultBaseURL = "https://api.kranth.com"
	userAgent      = "kranth-go/" + Version
)

// Client is the entry point.  One client per process is fine; the underlying
// net/http transport multiplexes goroutine-safely.
type Client struct {
	APIKey  string
	BaseURL string
	HTTP    *http.Client

	Sims     *SimsService
	APIKeys  *APIKeysService
	Models   *ModelsService
	Billing  *BillingService
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
	IdeaText       string `json:"idea_text"`
	PersonaCount   uint32 `json:"persona_count"`
	ModelID        string `json:"model_id"`
	TrainingOptIn  *bool  `json:"training_opt_in,omitempty"`
	IdempotencyKey string `json:"-"`
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
		req, err := http.NewRequestWithContext(ctx, "GET", s.c.BaseURL+"/v1/sims/"+url.PathEscape(simID)+"/events", nil)
		if err != nil {
			out <- StreamFrame{Err: err}
			return
		}
		req.Header.Set("Authorization", "Bearer "+s.c.APIKey)
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
