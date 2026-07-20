# Project Agent Rules

## Tailwind CSS v4 Configuration
- When initializing or writing configuration for Tailwind CSS, ALWAYS check the version.
- **Tailwind v4** does NOT use `tailwind.config.js`. 
- **Tailwind v4** uses `@import "tailwindcss";` in the main CSS file, replacing the old `@tailwind base;`, `@tailwind components;`, and `@tailwind utilities;` directives.
- When configuring Tailwind v4 for a **Vite** project, do not use PostCSS. Instead, install the new Vite plugin: `npm install tailwindcss @tailwindcss/vite` and add it to `vite.config.ts`. You do not need `postcss.config.js`.

## PyTorch Security
- When loading models or tensors using `torch.load()`, ALWAYS include the `weights_only=True` parameter to prevent arbitrary code execution vulnerabilities, especially when dealing with federated or external model weights.

## ML Pipeline Rules (SupplyShield / PyTorch projects)

- **Scaler must be fit only on training data.** Never call `scaler.fit()` or `scaler.fit_transform()` on the full dataset (including val/test). Always fit on `X_train` only, then `scaler.transform()` on val/test.

- **Calibrate classification threshold on imbalanced data.** Never hardcode `0.5` as the decision threshold when the dataset is imbalanced. After training, use Youden's J statistic on the validation set to find the optimal threshold: `threshold = thresholds[argmax(tpr - fpr)]`. Save the threshold to a checkpoint file and load it in the API.

## Auth & Multi-Tenant Database Rules (SupplyShield / SaaS projects)

- **Dual-DB pattern**: Use MongoDB for auth/identity (org registration, hashed passwords, org IDs). Use PostgreSQL for per-org analytical/transactional data (one DB per tenant).

- **Link field in MongoDB**: Every org document must have a `postgres_db` field pointing to the org's PostgreSQL database name. Never hardcode DB names in code — always resolve at runtime.

- **Org ID format**: Use `{PREFIX}-{uuid[:8].upper()}` e.g. `NVM-A1B2C3D4`. Show to the user only once at registration.

- **Password hashing**: Always use `passlib[bcrypt]` with `CryptContext(schemes=["bcrypt"])`. Pin `bcrypt==4.0.1` to avoid passlib compatibility issues.

- **JWT sessions**: Use `python-jose[cryptography]`. Store JWT as `supplyshield_jwt` in `localStorage`. All protected API calls send `Authorization: Bearer {jwt}`.

- **Seed script**: Create `backend/db/seed.py` as the idempotent bootstrap script. It must: (1) create Postgres DBs, (2) load CSV data into tables, (3) insert MongoDB org docs with hashed passwords.

- **Windows terminal encoding**: Always add `sys.stdout.reconfigure(encoding='utf-8')` at the top of any Python script that prints non-ASCII characters, to avoid `cp1252` errors on Windows.

