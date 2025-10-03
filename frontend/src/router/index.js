import { createRouter, createWebHistory } from 'vue-router'
import views from "@/views/index.js";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "",
      component: views.MapView,
    },
  ],
})

export default router
