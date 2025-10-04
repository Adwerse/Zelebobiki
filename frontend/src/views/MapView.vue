<script setup>
import { ref, onMounted } from 'vue';
import { Deck, TileLayer, BitmapLayer, MapView, ScatterplotLayer } from 'deck.gl';
import PopUp from './PopUp.vue' 
const deckContainer = ref(null);
const coords = ref(null);
let deck = ref(null);
onMounted(() => {
  if (!deckContainer.value) return;

  // Створюємо Deck
  


    const INITIAL_VIEW_STATE = {
      longitude: 0,
      latitude: 0,
      zoom: 2,
      pitch: 0,
      bearing: 0
    };

    const oceanBasemap = new TileLayer({
      id: 'ocean-basemap',
      data: 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      minZoom: 0,
      maxZoom: 13,
      tileSize: 256,

      renderSubLayers: props => {
        const {
          bbox: {west, south, east, north}
        } = props.tile;
        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north]
        });
      }
    });

    deck.value = new Deck({
      parent: deckContainer.value,
      initialViewState: INITIAL_VIEW_STATE,
      controller: true,
      views: [new MapView({repeat: true})],
      layers: [oceanBasemap],
      onClick: info => {
        console.log(info);
        if (info.coordinate) {
          coords.value = [info.x, info.y];
        }
      },
      onViewStateChange: ({viewState}) => {
        console.log('Карта рухається', viewState);
      }
    });
});
</script>

<template>
  <div>
    <div ref="deckContainer"></div>
    <!-- <PopUp
      v-if="coords"
      :deck="deck"
      :coordinates="coords"
      @close="coords = null"
    /> -->
  </div>
</template>
