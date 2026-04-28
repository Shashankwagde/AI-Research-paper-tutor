import textwrap

def get_pricing_html(active_plan="free"):
    # Monthly Buttons
    starter_btn_mo = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'starter' else '<a href="/?checkout=starter" target="_self" class="cta-button cta-standard">Get Started</a>'
    pro_btn_mo = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'pro' else '<a href="/?checkout=pro" target="_self" class="cta-button cta-primary">Upgrade to Pro</a>'
    elite_btn_mo = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'elite' else '<a href="/?checkout=elite" target="_self" class="cta-button cta-standard">Go Elite</a>'

    # Yearly Buttons
    starter_btn_yr = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'starter' else '<a href="/?checkout=starter&billing=yearly" target="_self" class="cta-button cta-standard">Get Started</a>'
    pro_btn_yr = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'pro' else '<a href="/?checkout=pro&billing=yearly" target="_self" class="cta-button cta-primary">Upgrade to Pro</a>'
    elite_btn_yr = '<a href="#" class="cta-button cta-disabled">Current Plan</a>' if active_plan == 'elite' else '<a href="/?checkout=elite&billing=yearly" target="_self" class="cta-button cta-standard">Go Elite</a>'

    return textwrap.dedent(f"""
    <div class="pricing-container">
        <input type="radio" id="billing-monthly" name="billing" checked style="display:none;">
        <input type="radio" id="billing-yearly" name="billing" style="display:none;">

        <div class="pricing-header">
            <h2>Supercharge Your Research</h2>
            <p>Upgrade to unlock advanced AI models, faster processing, and deeper analytical insights. Choose the plan that fits your research needs.</p>
            
            <div class="billing-toggle">
                <label for="billing-monthly" class="btn-monthly">Monthly</label>
                <label for="billing-yearly" class="btn-yearly">Yearly <span class="discount-badge">Save 20%</span></label>
            </div>
        </div>

        <div class="pricing-grid">
            <!-- STARTER PLAN -->
            <div class="pricing-card">
                <div class="plan-name">Starter</div>
                <div class="plan-price">
                    <span class="price-monthly">₹299</span>
                    <span class="price-yearly">₹239</span>
                    <span class="per-month">/month</span>
                </div>
                <p class="plan-desc">Perfect for students and casual researchers.</p>
                <div class="checkout-monthly">{starter_btn_mo}</div>
                <div class="checkout-yearly">{starter_btn_yr}</div>
                <div class="features-list">
                    <div class="feature-item"><span class="check">✓</span> Access to basic AI summary model</div>
                    <div class="feature-item"><span class="check">✓</span> 20 research paper summaries per month</div>
                    <div class="feature-item"><span class="check">✓</span> Standard summary generation speed</div>
                    <div class="feature-item"><span class="check">✓</span> Save summaries history</div>
                    <div class="feature-item"><span class="check">✓</span> Community support</div>
                    <div class="feature-item"><span class="check">✓</span> Limited access to free research papers</div>
                </div>
            </div>

            <!-- PRO RESEARCHER PLAN -->
            <div class="pricing-card popular">
                <div class="popular-badge">Most Popular</div>
                <div class="plan-name">Pro Researcher</div>
                <div class="plan-price">
                    <span class="price-monthly">₹799</span>
                    <span class="price-yearly">₹639</span>
                    <span class="per-month">/month</span>
                </div>
                <p class="plan-desc">For academics and professionals needing deep insights.</p>
                <div class="checkout-monthly">{pro_btn_mo}</div>
                <div class="checkout-yearly">{pro_btn_yr}</div>
                <div class="features-list">
                    <div class="feature-item"><span class="check">✓</span> Access to advanced AI models for deeper summaries</div>
                    <div class="feature-item"><span class="check">✓</span> 100 research paper summaries per month</div>
                    <div class="feature-item"><span class="check">✓</span> Faster processing speed</div>
                    <div class="feature-item"><span class="check">✓</span> AI-powered key insight extraction</div>
                    <div class="feature-item"><span class="check">✓</span> Chat with uploaded research papers</div>
                    <div class="feature-item"><span class="check">✓</span> Access to premium research paper library</div>
                    <div class="feature-item"><span class="check">✓</span> Priority email support</div>
                </div>
            </div>

            <!-- ELITE SCHOLAR PLAN -->
            <div class="pricing-card">
                <div class="plan-name">Elite Scholar</div>
                <div class="plan-price">
                    <span class="price-monthly">₹1499</span>
                    <span class="price-yearly">₹1199</span>
                    <span class="per-month">/month</span>
                </div>
                <p class="plan-desc">Unlimited power for intensive research labs and teams.</p>
                <div class="checkout-monthly">{elite_btn_mo}</div>
                <div class="checkout-yearly">{elite_btn_yr}</div>
                <div class="features-list">
                    <div class="feature-item"><span class="check">✓</span> Unlimited AI-generated summaries</div>
                    <div class="feature-item"><span class="check">✓</span> Access to the most powerful premium AI models</div>
                    <div class="feature-item"><span class="check">✓</span> Deep technical analysis of papers</div>
                    <div class="feature-item"><span class="check">✓</span> Unlimited research paper chat sessions</div>
                    <div class="feature-item"><span class="check">✓</span> Full access to premium & paid research papers</div>
                    <div class="feature-item"><span class="check">✓</span> Early access to new AI features</div>
                    <div class="feature-item"><span class="check">✓</span> Dedicated priority support</div>
                    <div class="feature-item"><span class="check">✓</span> Export summaries in PDF & DOCX</div>
                </div>
            </div>
        </div>
    </div>
    """)
