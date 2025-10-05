<script setup>
import { Button } from '@/components/ui/button/index.js';
import { Info, Map } from "lucide-vue-next";
import { RouterLink, useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()

const calculateMenuItemVariant = (path) => {
  return computed(() => {
    return route.path === path ? "default" : "secondary"
  })
}

</script>
<template>
  <header class="fixed left-0 top-0 w-full z-99 py-2.5">
    <div class="px-5 w-full">
      <div class="w-full flex justify-between items-center flex-wrap gap-x-10">
        <a href="/" class="text-xl flex items-center gap-2.5">
          <img src="@/assets/img/logo.png" alt="Logo" class="w-15">
          <span class="font-bold capitalize">Sharks from Space</span>
        </a>
        <ul class="flex gap-5">
          <li>
            <Button :as="RouterLink" to="/" :variant="calculateMenuItemVariant('/').value">
              <Map /> Interactive map
            </Button>
          </li>
          <li>
            <Button :variant="calculateMenuItemVariant('/about').value" :as="RouterLink" to="/about">
              <Info /> About
            </Button>
          </li>
        </ul>
      </div>
    </div>
  </header>
</template>
