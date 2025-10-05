<script setup>
import { ref } from 'vue'

// --------- Project quick facts (easy to tweak) ---------
const project = {
  title: 'Sharks from Space - Zelebobiki',
  tagline: 'Open tools to map, monitor, and learn about sharks from orbit.',
  challengeUrl: 'https://www.spaceappschallenge.org/2025/challenges/sharks-from-space/',
  teamUrl: 'https://www.spaceappschallenge.org/2025/find-a-team/zelebobiki/',
  repoUrl: 'https://github.com/Adwerse/Zelebobiki/tree/master',
  contactEmail: 'dmytrohryshchenkowork@gmail.com',
}

const highlights = [
  { label: 'Stack', value: 'Vue 3, Tailwind, deck.gl' },
  { label: 'Backend', value: 'Python + Django Rest Framework' },
  { label: 'Focus', value: 'Heatmaps, detection, education' },
]

const teammates = ref([
  {
    name: 'Dmytro Hryshchenko.',
    role: 'Fullstack Developer',
    photo:
      'https://i.ibb.co/k2qjhgyV/photo-2025-10-05-10-27-54.jpg',
    links: [
      { label: 'GitHub', href: 'https://github.com/h0pers' },
      { label: 'LinkedIn', href: 'https://www.linkedin.com/in/hryshchenko-dev/' },
    ],
  },
  {
    name: 'Kirill Tonkovyd',
    role: 'Map Research, Map Frontend',
    photo:
      'https://i.ibb.co/qLsSLsqq/photo-2025-10-05-10-27-27.jpg',
    links: [
      { label: 'GitHub', href: 'https://github.com/chillchirill' },
      { label: 'LinkedIn', href: 'https://www.linkedin.com/in/kirill-tonkovyd-226a60305/' },
    ],
  },
  {
    name: 'Rodion Omelich',
    role: 'AI / Dataset Engineer',
    photo:
      'https://i.ibb.co/F4rzyTmc/image.png',
    links: [
      { label: 'GitHub', href: 'https://github.com/RodionOm' },
      { label: 'LinkedIn', href: 'https://www.linkedin.com/in/rodion-omelich-bba55b378/' },
    ],
  },
  {
    name: 'Adam Vakar',
    role: 'AI / Dataset Engineer',
    photo:
      'https://i.ibb.co/RT3t2F4T/photo-2025-10-05-10-27-26.jpg',
    links: [
      { label: 'GitHub', href: 'https://github.com/Adwerse' },
      { label: 'LinkedIn', href: 'https://www.linkedin.com/in/adamvakar/' },
    ],
  },
])

// --------- Milestones (edit freely) ---------
const milestones = [
  { when: 'Day 1', what: 'Scaffold app, set up Tailwind, base map with deck.gl' },
  { when: 'Day 2', what: 'API prototype + heatmap endpoint' },
  { when: 'Day 3', what: 'UI polish, About page, docs, and demo scenario' },
]
</script>

<template>
  <div class="relative isolate">
    <!-- Hero -->
    <section
      class="relative overflow-hidden border-b border-white/10 bg-gradient-to-b from-sky-900 via-sky-950 to-black text-white"
    >
      <div
        class="absolute inset-0 opacity-30"
        :style="{
          backgroundImage:
            'url(https://images.unsplash.com/photo-1518837695005-2083093ee35b?q=80&w=1920&auto=format&fit=crop)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }"
        aria-hidden="true"
      />

      <div class="mx-auto max-w-7xl px-6 pt-30 pb-16 md:pt-32 md:pb-24">
        <div class="flex flex-col items-start gap-6 md:gap-8">
          <span
            class="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-1 text-sm backdrop-blur"
          >
            <span class="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-300" />
            NASA Space Apps 2025 — Team Zelebobiki
          </span>

          <h1 class="text-4xl font-bold leading-tight tracking-tight md:text-6xl">
            {{ project.title }}
          </h1>
          <p class="max-w-2xl text-lg text-white/80 md:text-xl">
            {{ project.tagline }}
          </p>

          <div class="flex flex-wrap items-center gap-3">
            <a
              :href="project.challengeUrl"
              target="_blank"
              class="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-slate-900 shadow hover:shadow-md focus:outline-none focus:ring-2 focus:ring-white/50"
            >
              <!-- icon: target -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor"><path d="M12 2a1 1 0 0 1 1 1v1.07A8.001 8.001 0 0 1 20.93 11H22a1 1 0 1 1 0 2h-1.07A8.001 8.001 0 0 1 13 20.93V22a1 1 0 1 1-2 0v-1.07A8.001 8.001 0 0 1 3.07 13H2a1 1 0 1 1 0-2h1.07A8.001 8.001 0 0 1 11 3.07V2a1 1 0 0 1 1-1Zm0 4a6 6 0 1 0 0 12A6 6 0 0 0 12 6Zm0 2a1 1 0 0 1 .993.883L13 9v2h2a1 1 0 0 1 .117 1.993L15 13h-2v2a1 1 0 0 1-1.993.117L11 15v-2H9a1 1 0 0 1-.117-1.993L9 11h2V9a1 1 0 0 1 1-1Z"/></svg>
              Challenge
            </a>
            <a
              :href="project.teamUrl"
              target="_blank"
              class="inline-flex items-center gap-2 rounded-xl border border-white/30 bg-white/5 px-4 py-2 shadow hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/50"
            >
              <!-- icon: users -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor"><path d="M7 10a4 4 0 1 1 0-8 4 4 0 0 1 0 8Zm10 0a4 4 0 1 1 0-8 4 4 0 0 1 0 8ZM7 12a5 5 0 0 1 5 5v2H2v-2a5 5 0 0 1 5-5Zm10 0a5 5 0 0 1 5 5v2h-6.5v-2a6.5 6.5 0 0 0-2.303-4.937A4.98 4.98 0 0 1 17 12Z"/></svg>
              Our Team
            </a>
            <a
              :href="project.repoUrl"
              target="_blank"
              class="inline-flex items-center gap-2 rounded-xl border border-white/30 bg-white/5 px-4 py-2 shadow hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white/50"
            >
              <!-- icon: code -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor"><path d="M8.707 17.293a1 1 0 0 1-1.414 1.414l-4-4a1 1 0 0 1 0-1.414l4-4a1 1 0 0 1 1.414 1.414L5.414 13l3.293 3.293Zm6.586 0 3.293-3.293-3.293-3.293a1 1 0 1 1 1.414-1.414l4 4a1 1 0 0 1 0 1.414l-4 4a1 1 0 1 1-1.414-1.414Z"/></svg>
              Source
            </a>
          </div>

          <div class="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div v-for="h in highlights" :key="h.label" class="rounded-2xl border border-white/10 bg-white/5 p-4 shadow">
              <p class="text-sm text-white/70">{{ h.label }}</p>
              <p class="mt-1 text-base font-medium">{{ h.value }}</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Mission / Approach -->
    <section class="mx-auto max-w-7xl px-6 py-12 md:py-16">
      <div class="grid items-start gap-10 md:grid-cols-2">
        <div class="space-y-4">
          <h2 class="text-2xl font-semibold md:text-3xl">Our mission</h2>
          <p class="text-slate-600">
            We build a lean, open, and educational toolset that helps people visualize shark presence
            using spaceborne data. Our goals: informed ocean literacy, transparent methods, and a
            demo that anyone can run.
          </p>
          <ul class="mt-4 list-disc space-y-2 pl-6 text-slate-700">
            <li>Heatmaps from aggregated observations (API).</li>
            <li>Deck.gl map for fast, fluid, multi‑scale exploration.</li>
            <li>Clear UX for students, scientists, and curious minds.</li>
          </ul>
        </div>
        <div>
          <img
            class="aspect-video w-full rounded-2xl object-cover shadow"
            src="https://cdn.pixabay.com/photo/2023/08/10/20/10/shark-8182315_1280.jpg"
            alt="Blue ocean with sun rays"
          />
          <p class="mt-2 text-xs text-slate-500">Photo © Pixabay</p>
        </div>
      </div>
    </section>

    <!-- How it works -->
    <section class="mx-auto max-w-7xl px-6 pb-8 md:pb-12">
      <div class="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm md:p-10">
        <h3 class="text-xl font-semibold md:text-2xl">How it works</h3>
        <div class="mt-6 grid gap-6 md:grid-cols-3">
          <div class="rounded-2xl border border-slate-200 p-5">
            <div class="text-sm font-medium text-slate-500">01 · Data</div>
            <p class="mt-2 text-slate-700">We collect shark‑related records and environmental layers and shape them into a simple API.</p>
          </div>
          <div class="rounded-2xl border border-slate-200 p-5">
            <div class="text-sm font-medium text-slate-500">02 · Map</div>
            <p class="mt-2 text-slate-700">The frontend renders a global basemap with deck.gl and overlays a heatmap from our endpoint.</p>
          </div>
          <div class="rounded-2xl border border-slate-200 p-5">
            <div class="text-sm font-medium text-slate-500">03 · Learn</div>
            <p class="mt-2 text-slate-700">Users explore patterns and read short, guided explanations right inside the app.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Team -->
    <section class="mx-auto max-w-7xl px-6 py-10 md:py-16">
      <div class="mb-6 flex items-end justify-between">
        <div>
          <h3 class="text-2xl font-semibold md:text-3xl">Meet the team</h3>
          <p class="mt-1 text-slate-600">Small, focused, and a little bit obsessed with clean UX and useful science.</p>
        </div>
        <a :href="project.teamUrl" target="_blank" class="text-sm font-medium text-sky-700 hover:underline">See official team page →</a>
      </div>
      <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <article
          v-for="mate in teammates"
          :key="mate.name"
          class="group rounded-3xl border border-slate-200 bg-white p-4 shadow-sm transition hover:-translate-y-1 hover:shadow-md"
        >
          <div class="overflow-hidden rounded-2xl">
            <img :src="mate.photo" :alt="mate.name" class="aspect-[4/3] w-full object-cover transition group-hover:scale-[1.03]" />
          </div>
          <div class="mt-4">
            <h4 class="text-lg font-semibold">{{ mate.name }}</h4>
            <p class="text-sm text-slate-600">{{ mate.role }}</p>
            <div class="mt-3 flex flex-wrap gap-2">
              <a
                v-for="l in mate.links"
                :key="l.href"
                :href="l.href"
                target="_blank"
                class="inline-flex items-center rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                {{ l.label }}
              </a>
            </div>
          </div>
        </article>
      </div>
    </section>

    <!-- Milestones -->
    <section class="mx-auto max-w-7xl px-6 pb-12 md:pb-20">
      <div class="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm md:p-10">
        <h3 class="text-xl font-semibold md:text-2xl">Hackathon milestones</h3>
        <ol class="mt-6 grid gap-4 md:grid-cols-3">
          <li v-for="m in milestones" :key="m.when" class="rounded-2xl border border-slate-200 p-5">
            <div class="text-sm font-medium text-slate-500">{{ m.when }}</div>
            <p class="mt-2 text-slate-700">{{ m.what }}</p>
          </li>
        </ol>
      </div>
    </section>

    <!-- Call to action -->
    <section class="mx-auto max-w-7xl px-6 pb-16">
      <div class="flex flex-col items-center gap-4 rounded-3xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <h3 class="text-2xl font-semibold">Collaborate or learn more</h3>
        <p class="max-w-2xl text-slate-600">
          Questions, ideas, or datasets to share? We’d love to hear from you. This project is open and
          welcoming—perfect for students and ocean‑curious developers.
        </p>
        <div class="flex flex-wrap items-center justify-center gap-3">
          <a :href="project.repoUrl" target="_blank" class="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2 font-medium text-white shadow hover:bg-sky-700">
            Open the repo
          </a>
          <a :href="`mailto:${project.contactEmail}`" class="inline-flex items-center gap-2 rounded-xl border border-slate-300 px-4 py-2 font-medium text-slate-800 hover:bg-slate-50">
            Email us
          </a>
        </div>
      </div>
      <p class="mt-4 text-center text-xs text-slate-500">
        Shark & ocean images via Unsplash open links. Please replace placeholders with your own assets and credits as needed.
      </p>
    </section>
  </div>
</template>

<style scoped>
/***** Optional: soft noise overlay for hero *****/
.hero-noise::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.05;
  background-image: url('data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'160\' height=\'160\'><filter id=\'n\'><feTurbulence type=\'fractalNoise\' baseFrequency=\'0.65\' numOctaves=\'3\'/><feColorMatrix type=\'saturate\' values=\'0\'/></filter><rect width=\'100%\' height=\'100%\' filter=\'url(%23n)\'/></svg>');
  mix-blend-mode: soft-light;
}
</style>
