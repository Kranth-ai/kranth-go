# Kranth — Go SDK

```bash
go get github.com/Kranth-ai/kranth-go
```

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Kranth-ai/kranth-go"
)

func main() {
	ctx := context.Background()
	client := kranth.New("kr_live_…")

	sim, err := client.Sims.Create(ctx, kranth.CreateSimRequest{
		IdeaText:     "We're launching a paid Rust devtool. $29/mo, hosted, no self-host.",
		PersonaCount: 50,
		ModelID:      "claude-sonnet-4-6",
	})
	if err != nil {
		log.Fatal(err)
	}

	for frame := range client.Sims.Stream(ctx, sim.SimID) {
		if frame.Err != nil {
			log.Fatal(frame.Err)
		}
		switch frame.Event.Kind {
		case "persona.ready":
			fmt.Println("persona ready")
		case "reaction.complete":
			fmt.Println("reaction complete")
		case "sim.complete":
			fmt.Println("verdict:", string(frame.Event.Data))
			return
		}
	}
}
```

## Resources

- `client.Sims.Create(ctx, kranth.CreateSimRequest{...})`
- `client.Sims.Get(ctx, simID)`
- `client.Sims.List(ctx, kranth.ListSimsParams{Status: "running", Limit: 50})`
- `client.Sims.Stream(ctx, simID)` — `<-chan StreamFrame`
- `client.Sims.Cancel(ctx, simID)`
- `client.Sims.Export(ctx, simID)`
- `client.Recon.Create(ctx, kranth.CreateReconRequest{...})` / `.Get` / `.List` / `.Tiers` / `.Export` / `.Stream` — web-grounded research swarm
- `client.Debates.Create(ctx, kranth.CreateDebateRequest{...})` / `.Get` / `.List` / `.Turns` / `.SetPublic` / `.Cancel` / `.Export` / `.Stream` — adversarial panel
- `client.APIKeys.Create(ctx, name, env)` / `.List(ctx)` / `.Revoke(ctx, id)`
- `client.Models.List(ctx)`
- `client.Billing.Usage(ctx)` / `.CheckoutURL(ctx, params)` / `.PortalURL(ctx, returnURL)`
- `client.Me(ctx)`

## Errors

```go
if _, err := client.Sims.Create(ctx, req); err != nil {
    switch {
    case kranth.IsPaymentRequired(err):
        // out of credits — top up
    case kranth.IsRateLimited(err):
        // var apiErr *kranth.APIError; errors.As(err, &apiErr)
        // back off for *apiErr.RetryAfter seconds
    }
}
```

## Configuration

```go
client := kranth.New(
    "kr_live_…",
    kranth.WithBaseURL("https://api.kranth.ai"),
    kranth.WithHTTPClient(&http.Client{Timeout: 60 * time.Second}),
)
```

## License

Apache-2.0
